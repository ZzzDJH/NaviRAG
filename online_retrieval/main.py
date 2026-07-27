from retrieval_pipeline import *
from LLM import chatllm
from tqdm import tqdm
import json
import argparse
import os
from sentence_transformers import SentenceTransformer
import faiss
import time
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent


def load_data(data_path:str):
    queries,answers,titles = [],[],[]
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line.strip())
            queries.append(obj.get("query", ""))
            answers.append(obj.get("answer", ""))
            titles.append(obj.get("title", ""))
    return queries,answers,titles

def load_data_loogle(data_path:str):
    queries,answers,titles,evis = [],[],[],[]
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line.strip())
            queries.append(obj.get("query", ""))
            answers.append(obj.get("answer", ""))
            titles.append(obj.get("title", ""))
            evis.append(obj.get("evidence", ""))
    return queries,answers,titles,evis

def load_data_lbv2(data_path: str):
   
    queries = []
    answers = []
    titles = []
    sub_domains = []
    difficulties = []
    lengths = []

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())

            query = obj.get("query", "")
            answer = obj.get("answer", "")
            title = obj.get("title", "")
            choice_a = obj.get("choice_A", "")
            choice_b = obj.get("choice_B", "")
            choice_c = obj.get("choice_C", "")
            choice_d = obj.get("choice_D", "")

            full_query = (
                f"{query}\n"
                f"A. {choice_a}\n"
                f"B. {choice_b}\n"
                f"C. {choice_c}\n"
                f"D. {choice_d}"
            )

            sub_domain = obj.get("sub_domain", "")
            difficulty = obj.get("difficulty", "")
            length = obj.get("length", "")

            queries.append(full_query)
            answers.append(answer)
            titles.append(title)
            sub_domains.append(sub_domain)
            difficulties.append(difficulty)
            lengths.append(length)

    return (
        queries,
        answers,
        titles,
        sub_domains,
        difficulties,
        lengths,
    )


def load_wiki_data(wiki_path:str):
 
    with open(wiki_path, "r", encoding="utf-8") as f:
        wiki_data = json.load(f)
    return wiki_data


def load_index(index_path: str):

    index = faiss.read_index(index_path)
    
    return index

def save_predictions_jsonl(
    predictions: list[str],
    contexts_list: list[str],
    save_path: str
):

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for prediction, context in zip(predictions, contexts_list):
            item = {
                "prediction": prediction,
                "context": context
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(predictions)} results to {save_path}")

def main(
        wRAG_line:str,
        qa_data_path:str,
        data_dir:str="",
        save_path:str="",

        embedding_model:str="", 
        is_vllm:bool=True,
        vllm_model_path:str="", 
        
        top_k:int=5, 
        dataset:str="", 
        prompt_template_path:str="",
        
        is_lc:bool=True,
        max_tokens=5120,

        ):

    num_wiki_qa = 0

    # Step 1
    if dataset == "loogle":
        queries, answers,titles,evis = load_data_loogle(qa_data_path)
    elif dataset == "narrative":
        queries, answers,titles = load_data(qa_data_path)
    else:#longbench-v2
        queries, answers,titles,sub_domains,difficulties,lengths = load_data_lbv2(qa_data_path)

    chunk_path_v = os.path.join(data_dir, "v_chunks.json")
    index_path_v = os.path.join(data_dir, "v_index.faiss")
    chunk_leaf_path = os.path.join(data_dir, "chunks.json")
    wiki_ori_path = os.path.join(data_dir, "ud_wiki.jsonl")
    segment_path = os.path.join(data_dir, "segments.jsonl")
    chunk_sum_path = os.path.join(data_dir, "sum.json")


    # Step 2
    llmmodel = chatllm(is_vllm=is_vllm, model_name=vllm_model_path)
    embedder = SentenceTransformer(embedding_model)


    wiki_data_v = load_wiki_data(chunk_path_v)
    faiss_id_to_chunk_v = {
        int(k): v for k, v in wiki_data_v.items()
    } 

    index_v = load_index(index_path_v)

    with open(prompt_template_path, "r") as f:
        wiki_prompt_json = json.load(f)


    print("initialized")
    # Step 3
    start_time = time.time()  
    if wRAG_line == "vanilla":
        predictions = []
        seen_ids_list = []
        contexts_list = []
        all_refs_len = []

        for query, title in tqdm(zip(queries, titles), total=len(queries), desc="Processing Queries"):
            prediction,seen_ids,context,num_tokens = vanilla(
                query=query,
                title=title,
                model=llmmodel, 
                faiss_id_to_chunk_vanilla=faiss_id_to_chunk_v,
                index_vanilla=index_v, 
                embeder=embedder,      
                is_lc=is_lc,      
            
                top_k=top_k, 
                dataset=dataset, 
                wiki_prompt_json=wiki_prompt_json,
                
                )

            seen_ids_list.append([str(i) for i in seen_ids])
            predictions.append(prediction)
            contexts_list.append(context)
            all_refs_len.append(num_tokens)

    elif wRAG_line == "navirag": 

        wiki_data_f = load_wiki_data(chunk_leaf_path)
        faiss_id_to_chunk_f = {
            k: v for k, v in wiki_data_f.items()
        } 

        wiki_data_s = load_wiki_data(chunk_sum_path)
        sum_chunks = {
            int(k): v for k, v in wiki_data_s.items()
        } 

        predictions = []
        seen_ids_list = []
        all_sum_len = []
        all_refs_len = []
        count_list = []
        contexts_list = []
        for query, title in tqdm(zip(queries, titles), total=len(queries), desc="Processing Queries"):
            prediction,seen_ids, sum_len, has_wiki,num_tokens,count,context = navirag( 
                query=query,
                title=title,
                model=llmmodel, 
                faiss_id_to_chunk_vanilla=faiss_id_to_chunk_v,
                index_vanilla=index_v, 
                faiss_id_to_chunk_leaf=faiss_id_to_chunk_f,
                embeder=embedder,      
                is_lc=is_lc,      
            
                top_k=top_k, 
                dataset=dataset, 
                wiki_prompt_json=wiki_prompt_json,

                wiki_ori_path =wiki_ori_path,
                segment_path = segment_path,

                sum_chunks = sum_chunks,
                max_tokens= max_tokens
                )

            seen_ids_list.append([str(i) for i in seen_ids])
            predictions.append(prediction)
            all_sum_len.append(sum_len)
            if has_wiki:
                num_wiki_qa += 1
            all_refs_len.append(num_tokens)
            count_list.append(count)
            contexts_list.append(context)

    elif wRAG_line == "note": 

        wiki_data_f = load_wiki_data(chunk_leaf_path)
        faiss_id_to_chunk_f = {
            k: v for k, v in wiki_data_f.items()
        } 

        wiki_data_s = load_wiki_data(chunk_sum_path)
        sum_chunks = {
            int(k): v for k, v in wiki_data_s.items()
        } 

        predictions = []
        seen_ids_list = []
        all_sum_len = []
        all_refs_len = []
        count_list = []
        contexts_list = []
        note_contexts_list = []
        note_num = 0
        for query, title in tqdm(zip(queries, titles), total=len(queries), desc="Processing Queries"):
            prediction,seen_ids, sum_len, has_wiki,num_tokens ,count,context,note,is_note= navirag_note( 
                query=query,
                title=title,
                model=llmmodel, 
                faiss_id_to_chunk_vanilla=faiss_id_to_chunk_v,
                index_vanilla=index_v, 
                faiss_id_to_chunk_leaf=faiss_id_to_chunk_f,
                embeder=embedder,      
                is_lc=is_lc,      
            
                top_k=top_k, 
                dataset=dataset, 
                wiki_prompt_json=wiki_prompt_json,

                wiki_ori_path =wiki_ori_path,
                segment_path = segment_path,


                sum_chunks = sum_chunks,
                max_tokens= max_tokens
                )

            seen_ids_list.append([str(i) for i in seen_ids])
            predictions.append(prediction)
            all_sum_len.append(sum_len)
            if has_wiki:
                num_wiki_qa += 1
            all_refs_len.append(num_tokens)
            count_list.append(count)
            contexts_list.append(context)
            note_contexts_list.append(note)
            if is_note:
                note_num += 1
            
    else:
        raise ValueError("Invalid wRAG_line value")

    end_time = time.time()  
    print(f"running time: {end_time - start_time:.4f} s")

    save_predictions_jsonl(
        predictions,
        contexts_list,
        save_path
    )


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--wRAG_line", required=True)
    parser.add_argument("--qa_data_path", required=True)
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing all retrieval and generation data files")
    parser.add_argument("--save_path", type=str, required=True)

    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="narrative") 
    parser.add_argument("--vllm_model_path", type=str, default="/path/to/llama33_70b")
    parser.add_argument("--embedding_model", type=str, default="/path/to/bge-m3")

    parser.add_argument("--prompt_template_path", type=str, default= SCRIPT_DIR / "prompts.json")  

    parser.add_argument(
        "--is_vllm",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--is_lc",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--max_tokens", type=int, default=8192)


    args = parser.parse_args()

    main(**vars(args))