import asyncio
import faiss
import numpy as np
from typing import List, Dict,Tuple,Any, Set, Optional
import json
from difflib import SequenceMatcher
import re

def build_title_subindex(
    title: str,
    full_index: faiss.IndexFlatIP,
    chunk_map: dict
) -> tuple[faiss.IndexFlatIP, dict, dict]:
    

    matched_ids = [
        int(i) for i, data in chunk_map.items()
        if isinstance(data.get("path"), list) and data["path"] and data["path"][0] == title
    ]

    if not matched_ids:
        raise ValueError(f"[build_title_subindex] No chunks found for title: {title}")

    vectors = np.vstack([full_index.reconstruct(i) for i in matched_ids])

    sub_index = faiss.IndexFlatIP(vectors.shape[1])
    sub_index.add(vectors)

    local2global_map = {local: matched_ids[local] for local in range(len(matched_ids))}
    sub_chunk_map = {str(local): chunk_map[matched_ids[local]] for local in range(len(matched_ids))}

    return sub_index, local2global_map, sub_chunk_map


def get_unique_faiss_ids(
    query_vec,
    index,
    seen_faiss_ids: set,
    base_topk: int
) -> tuple[list[int], set[int]]:
    
    raw_topk = base_topk + len(seen_faiss_ids)
    if raw_topk >= index.ntotal:
        print("Insufficient remaining vectors, skip vanilla retrieval.")
        return [], seen_faiss_ids

    distances, indices = index.search(query_vec, raw_topk)



    filtered_ids = []
    for idx in indices[0]:
        if idx == -1:
            continue  
        if idx not in seen_faiss_ids:
            seen_faiss_ids.add(idx)
            filtered_ids.append(idx)
        if len(filtered_ids) >= base_topk:
            break

    return filtered_ids, seen_faiss_ids



def LLM_final(
    model,
    query: str,
    context:str,
    dataset: str = " ",
    wiki_prompt_json = None
) -> str:
 
    prompt_template = wiki_prompt_json[dataset]["final_answer0"]

    prompt = prompt_template.format(query=query, context=context)

    prediction = model.generate(prompt)

    return prediction.strip()
    

def remove_empty_path_chunks(sum_chunks):
    return {
        idx: data for idx, data in sum_chunks.items()
        if isinstance(data.get("path"), list) and len(data["path"]) > 0
    }

def filter_chunks_by_title(sum_chunks, title):
    return {
        idx: data for idx, data in sum_chunks.items()
        if data["path"] and data["path"][0] == title
    }

def extract_root_paths(
    retrieved_paths: List[List[str]],
    root_n: int
) -> List[List[str]]:
    
    output_paths = []
    seen_roots: Set[str] = set()

    for path in retrieved_paths:
        if len(path) <= 1:
            continue  

        if len(path) > root_n:
            sub_path = path[:root_n + 1]
            root = path[root_n]
        else:
            sub_path = path[:-1] if len(path) >= 2 else path
            root = path[-2] if len(path) >= 2 else path[-1]

        if root in seen_roots:
            continue
        seen_roots.add(root)

        output_paths.append(sub_path)

    return output_paths

def ids_to_summary_list(sum_list, sub_sum_chunks):
    dedup_sum_list = list(dict.fromkeys(sum_list))
    sum_text_list = [
        sub_sum_chunks[chunk_id]["summary"]
        for chunk_id in dedup_sum_list
        if chunk_id in sub_sum_chunks
    ]
    return dedup_sum_list, sum_text_list

def note_query_judge(query, model, dataset, wiki_prompt_json):
    
    prompt_template = wiki_prompt_json[dataset]["query_note_judge3"]
    prompt = prompt_template.format(question=query)

    response = model.generate(prompt).strip()

    judge_res = parse_query_judgement(response)

    return judge_res

def parse_query_judgement(model_output: str) -> bool:
    
    if not model_output:
        return False

    output = model_output.strip().upper()

    return output == "B"


def load_sub_tree_from_path(path: List[str], wiki_file_path: str) -> Dict[str, Any]:
    
    with open(wiki_file_path, 'r', encoding='utf-8') as f:
        wiki_data = json.load(f)

    
    
    if not path:
       return wiki_data

    current = wiki_data.get(path[0])
    if current is None:
        raise KeyError(f"根节点 '{path[0]}' 不存在于 Wiki 数据中")

    for key in path[1:]:
        if key not in current:
            raise KeyError(f"路径中的节点 '{key}' 不存在于当前子树中")
        current = current[key]

    return current

def filter_and_trim_paths(sub_sum_chunks, root_list):
    
    filtered = {}

    for idx, data in sub_sum_chunks.items():
        path = data.get("path", [])
        if path[:len(root_list)] == root_list:
            remaining_path = path[len(root_list):]
            if remaining_path:

                new_path = remaining_path
                filtered[idx] = {
                    "path": new_path,
                    "summary": data["summary"]
                }

    return filtered

def note_refine(note, text_list, query, model, dataset, wiki_prompt_json,tokenizer):
    
    refs = "\n".join(text_list)

    prompt_template_j = wiki_prompt_json[dataset]["ref_judge_1"]
    prompt_j = prompt_template_j.format(query=query, documents=refs,notes=note)

    judge_res = model.generate(prompt_j).strip()

    judgement = parse_llm_judgement(judge_res)

    if judgement == "yes":

        prompt_template_n = wiki_prompt_json[dataset]["note_dj2"]
        prompt_n = prompt_template_n.format(query=query, documents=refs,notes=note )

        note_new = model.generate(prompt_n).strip()
    else:
        note_new = note

    return note_new


def parse_llm_judgement(text: str) -> str:
    if not text:
        return "no"

    cleaned = text.strip().lower()

    if cleaned == "yes":
        return "yes"

    cleaned2 = "".join(ch for ch in cleaned if ch.isalpha())

    if cleaned2 == "yes":
        return "yes"

    return "no"


def leaf_select_parallel(
    text: str,
    query: str,
    search_index_list: List[str],
    model,
    dataset,
    wiki_prompt_json
    ) -> Tuple[List[str], List[str]]:

    lines = split_text_by_index(text)

    entries = []
    for i, line in enumerate(lines):
        index_matches = re.findall(r"<(.*?)>", line)
        content = re.sub(r"<.*?>", "", line).strip()
        if content and index_matches:
            entries.append((i, content, index_matches))
        else:
            print(f"⚠️ formatting error or a missing index, skip this line: {line}")

    if not entries:
        return search_index_list

    texts_for_prompt = "\n".join([f"[{i}] {text}" for i, text, _ in entries])

    prompt_template = wiki_prompt_json[dataset]["leaf_select_new_old"]
    prompt = prompt_template.format(query=query, texts=texts_for_prompt)
    prediction = model.generate(prompt).strip()

    if not prediction or prediction.lower() == "none":
        return search_index_list

    try:
        selected_serials = [p.strip("[] ").strip() for p in prediction.split("//") if p.strip()]
        selected_serials = [int(p) for p in selected_serials if p.isdigit()]
    except Exception as e:
        print(f"⚠️ The filtering result parsing failed. Original output: {prediction}")
        return search_index_list

    for serial in selected_serials:
        if 0 <= serial < len(entries):
            _, selected_text, selected_indices = entries[serial]

            added = False  

            for idx in selected_indices:
                if idx.isdigit():
                    search_index_list.append(idx)
                    added = True

                else:
                    print(f"⚠️ Skip non-numeric indexes:{idx}")


        else:
            print(f"⚠️ Invalid serial number:{serial}")

    return search_index_list

def split_text_by_index(text: str) -> List[str]:

    raw_lines = text.strip().split('\n')
    merged_lines = []
    buffer = ""

    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        buffer += (" " if buffer else "") + line
        if re.search(r"<\d+>\s*$", line):  
            merged_lines.append(buffer.strip())
            buffer = ""

    if buffer:
        merged_lines.append(buffer.strip())

    return merged_lines

def get_by_path(d, path):
    current = d
    for key in path:
        if key not in current:
            print(f"[DEBUG] This key does not exist in the current dictionary: {key}")
            print(f"[DEBUG] Currently available keys: {list(current.keys())}")
            raise KeyError(key)
        current = current[key]
    return current  


def read_select_titles_parallel(root,path, titles,sum_list, query,  model, dataset, subtree_sum_chunks,wiki_prompt_json):
        
    if path:
        all_paths = root + '//' + '//'.join(path)
    else:
        all_paths = root

    title_summary_map = {}  

    full_prefix = path 
    prefix_len = len(full_prefix)

    for chunk_id, data in subtree_sum_chunks.items():
        chunk_path = data.get("path", [])
        
        if path == []:
            if len(chunk_path) == 1:
                title = chunk_path[0]
                if title in titles:
                    title_summary_map[title] = (data["summary"], chunk_id)
        else:
            if chunk_path[:prefix_len] == full_prefix:
                remaining = chunk_path[prefix_len:]
                if len(remaining) == 1:
                    title = remaining[0]
                    if title in titles:
                        title_summary_map[title] = (data["summary"], chunk_id)


    outlines = "\n".join(
        f"{i+1}. {title}: {clean_summary(title_summary_map.get(title, ('', None))[0])}"
        for i, title in enumerate(titles)
    )
    
    prompt_template = wiki_prompt_json[dataset]["t7next_9thexp"]
    prompt = prompt_template.format(entries=outlines, path=all_paths,question=query)

    
    response = model.generate(prompt) 
    response = response.strip()

    selected_titles, sum_titles = parse_titles_from_response_sum_2nd(response,titles) 

    if sum_titles:
        new_sum_list = [
            title_summary_map.get(title, ('', None))[1]
            for title in sum_titles
            if title in title_summary_map
        ]
        sum_list.extend(new_sum_list)

    return selected_titles, sum_list

def clean_summary(text: str) -> str:
    return text.replace("\n", " ").replace("\r", " ").strip()


def parse_titles_from_response_sum_2nd(response: str,
                                   valid_titles: List[str],
                                   sim_threshold: float = 0.9) -> Optional[List[str]]:
    
    response = response.strip()
    if response.lower() == "none":
        return None,None

    selected_titles = []  
    sum_titles = []      

    for line in response.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split("//")

        if len(parts) < 2:
            print(f"[WARN] Invalid line format: '{line}'")
            continue

        index_str = parts[0].strip()
        title_str = parts[1].strip()
        category = parts[2].strip().upper() if len(parts) >= 3 else "EXPLORE"

        if category not in {"EXPLORE", "INFO", "BOTH"}:
            print(f"[WARN] Unknown category '{category}', Skip this line: {line}")
            continue

        matched_title = None
        try:
            index = int(index_str) - 1
            if 0 <= index < len(valid_titles):
                matched_title = valid_titles[index]
            else:
                print(f"[WARN] Index {index+1} out of range, attempt fallback matching '{title_str}'")
        except ValueError:
            print(f"[WARN] Unable to resolve index: '{index_str}', Attempt fallback matching '{title_str}'")

        if not matched_title:
            best_match, best_score = None, 0.0
            for cand in valid_titles:
                score = sim_matching(title_str, cand)
                if score > best_score:
                    best_match, best_score = cand, score

            if best_match and best_score >= sim_threshold:
                print(f"[SIM_MATCH] replace:'{title_str}' → '{best_match}' ({best_score:.2f})")
                matched_title = best_match
            else:
                print(f"[FALLBACK] Unmatched:'{title_str}' Skipped")
                continue

        if category in {"EXPLORE", "BOTH"}:
            selected_titles.append(matched_title)

        if category in {"INFO", "BOTH"}:
            sum_titles.append(matched_title)

    selected_titles = list(dict.fromkeys(selected_titles))
    sum_titles = list(dict.fromkeys(sum_titles))

    if not selected_titles:
        selected_titles = None
    if not sum_titles:
        sum_titles = None

    return selected_titles, sum_titles

def sim_matching(title: str, candidate: str) -> float:
    return SequenceMatcher(None, title.strip(), candidate.strip()).ratio()


def leaf_select_note_step(
    text: str,
    query: str,
    search_index_list: List[str],
    wiki_seen_ids: List[str],
    search_text_list: List[str],
    note,
    is_lc,
    local2global_map_v,
    sub_chunk_map_v,
    model,
    dataset,
    wiki_prompt_json,
    tokenizer
    ) -> Tuple[List[str], List[str]]:


    lines = split_text_by_index(text)


    entries = []
    for i, line in enumerate(lines):

        index_matches = re.findall(r"<(.*?)>", line)

        content = re.sub(r"<.*?>", "", line).strip()
        if content and index_matches:
            entries.append((i, content, index_matches))
        else:
            print(f"⚠️ Skip the line if it has a formatting error or a missing index: {line}")

    if not entries:
        return search_index_list,wiki_seen_ids, search_text_list,note


    texts_for_prompt = "\n".join([f"[{i}] {text}" for i, text, _ in entries])


    prompt_template = wiki_prompt_json[dataset]["leaf_select_new_old"]
    prompt = prompt_template.format(query=query, texts=texts_for_prompt)
    prediction = model.generate(prompt).strip()

    if not prediction or prediction.lower() == "none":
        return search_index_list,wiki_seen_ids, search_text_list,note


    try:
        selected_serials = [p.strip("[] ").strip() for p in prediction.split("//") if p.strip()]
        selected_serials = [int(p) for p in selected_serials if p.isdigit()]
    except Exception as e:
        print(f"⚠️ The filtering result parsing failed. Original output: {prediction}")
        return search_index_list,wiki_seen_ids, search_text_list,note


    current_search_list = []
    for serial in selected_serials:
        if 0 <= serial < len(entries):
            _, selected_text, selected_indices = entries[serial]

            added = False  

            for idx in selected_indices:
                if idx.isdigit():
                    if idx not in wiki_seen_ids:
                        search_index_list.append(idx)
                        wiki_seen_ids.append(idx)
                        added = True

                        current_search_list.append(idx)

                    
                    
                else:
                    print(f"⚠️ Skip non-numeric indexes:{idx}")

            if added:
                search_text_list.append(selected_text)  

        else:
            print(f"⚠️ Invalid serial number:{serial}")

    if current_search_list:
        search_global_ids = set(int(x) for x in current_search_list)
        if is_lc:
            global2local_map_v = {v: k for k, v in local2global_map_v.items()}
            search_local_ids = {global2local_map_v[gid] for gid in search_global_ids if gid in global2local_map_v}
        else:
            search_local_ids = search_global_ids
        current_texts = [sub_chunk_map_v[str(i)]["text"] for i in search_local_ids]

        note = note_refine(note,current_texts,query,model,dataset,wiki_prompt_json,tokenizer)

    return search_index_list,wiki_seen_ids, search_text_list,note


def read_select_titles_with_note(root, path, titles, sum_list, query, note, model, dataset,  subtree_sum_chunks, wiki_prompt_json, tokenizer,batch_size=5):
    if path:
        all_paths = root + '//' + '//'.join(path)
    else:
        all_paths = root


    title_summary_map = {}
    full_prefix = path
    prefix_len = len(full_prefix)

    for data in subtree_sum_chunks.values():
        chunk_path = data.get("path", [])
        if path == []:
            if len(chunk_path) == 1:
                title = chunk_path[0]
                if title in titles:
                    title_summary_map[title] = data["summary"]
        else:
            if chunk_path[:prefix_len] == full_prefix:
                remaining = chunk_path[prefix_len:]
                if len(remaining) == 1:
                    title = remaining[0]
                    if title in titles:
                        title_summary_map[title] = data["summary"]


    available_titles = [t for t in titles if t in title_summary_map]
    if not available_titles:
        return [], sum_list, note  


    all_selected_titles = []
    all_sum_titles = []

    for i in range(0, len(available_titles), batch_size):
        batch_titles = available_titles[i:i + batch_size]


        outlines = "\n".join(
            f"{idx+1}. {title}: {clean_summary(title_summary_map[title])}"
            for idx, title in enumerate(batch_titles)
        )


        prompt_template = wiki_prompt_json[dataset]["9e_note1"]

        prompt = prompt_template.format(
            entries=outlines,
            path=all_paths,
            question=query,
            note=note if note.strip() else ""  
        )


        response = model.generate(prompt).strip()


        selected_batch, sum_batch = parse_titles_from_response_sum_2nd(response, batch_titles)

        if selected_batch:
            all_selected_titles.extend(selected_batch)
        if sum_batch:
            all_sum_titles.extend(sum_batch)



    if all_sum_titles:
        new_sum_list = [clean_summary(title_summary_map[title]) for title in all_sum_titles]
        sum_list.extend(new_sum_list)
        note = note_refine(note, new_sum_list, query, model, dataset, wiki_prompt_json,tokenizer)

    return all_selected_titles, sum_list, note
