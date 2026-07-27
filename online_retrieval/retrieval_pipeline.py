from typing import List
from utiles import *
from navigation import process_all_subtrees_deep_parallel, read_search_note_step


def vanilla(query:str,
            title:str,
            model, 
            faiss_id_to_chunk_vanilla:dict,
            index_vanilla, 
            embeder,      
            is_lc:bool,      
        
            top_k:int, 
            dataset:str=" ", 
            wiki_prompt_json=None,
            
            
)->str: 

    seen_ids = set()
    
    if is_lc: 
        sub_index_v, local2global_map_v, sub_chunk_map_v = build_title_subindex(
            title=title,
            full_index=index_vanilla,
            chunk_map=faiss_id_to_chunk_vanilla
        )   
    else:
        sub_index_v = index_vanilla
        sub_chunk_map_v = {str(i): v for i, v in faiss_id_to_chunk_vanilla.items()}

    query_vec = embeder.encode(
            query,
            convert_to_numpy=True,           
            normalize_embeddings=True      
            )


    query_vec = query_vec.reshape(1, -1) 

    new_ids, seen_ids = get_unique_faiss_ids(
        query_vec=query_vec,
        index=sub_index_v,
        seen_faiss_ids=seen_ids,
        base_topk=top_k
    )


    retrieved_wikis = [sub_chunk_map_v[str(i)] for i in new_ids]
    retrieved_texts = [item["text"] for item in retrieved_wikis]

    refs = "\n".join(retrieved_texts)

    if is_lc:
        seen_ids = set(local2global_map_v[i] for i in seen_ids)
        
    num_tokens = len(embeder.tokenizer.tokenize(refs))
    prediction = LLM_final(model=model, query=query, context=refs, dataset=dataset, wiki_prompt_json=wiki_prompt_json)

    return prediction,seen_ids, refs,num_tokens


def navirag(query:str,
            title:str,
            model, 
            faiss_id_to_chunk_vanilla:dict,
            index_vanilla, 
            embeder,      
            is_lc:bool,      
            faiss_id_to_chunk_leaf:dict,

            top_k:int, 
            dataset:str=" ", 
            wiki_prompt_json=None,
            
            wiki_ori_path="",
            segment_path="",

            sum_chunks = None,
            max_tokens= 5120
            
)->str: 
    
    count = 0

    has_wiki = True

    seen_ids = set()
    
    if is_lc: 
        sub_index_v, local2global_map_v, sub_chunk_map_v = build_title_subindex(
            title=title,
            full_index=index_vanilla,
            chunk_map=faiss_id_to_chunk_vanilla
        )   
        cleaned_chunks = remove_empty_path_chunks(sum_chunks)

        sub_sum_chunks = filter_chunks_by_title(cleaned_chunks, title) 
    else:
        sub_index_v = index_vanilla
        sub_chunk_map_v = {str(i): v for i, v in faiss_id_to_chunk_vanilla.items()}
        sub_sum_chunks = remove_empty_path_chunks(sum_chunks)

    query_vec = embeder.encode(
            query,
            convert_to_numpy=True,           
            normalize_embeddings=True      
            )


    query_vec = query_vec.reshape(1, -1) 

    new_ids, seen_ids = get_unique_faiss_ids(
        query_vec=query_vec,
        index=sub_index_v,
        seen_faiss_ids=seen_ids,
        base_topk=top_k
    )

    retrieved_wikis = [sub_chunk_map_v[str(i)] for i in new_ids]
    retrieved_texts = [item["text"] for item in retrieved_wikis]

    refs1 = "\n".join(retrieved_texts)

    if is_lc:
        global_ids = set(local2global_map_v[i] for i in seen_ids)
        wiki_seen_ids = [str(x) for x in global_ids]
    else:
        wiki_seen_ids = [str(x) for x in seen_ids]

    all_wiki_indices = [] 
    for item in retrieved_wikis:
        all_wiki_indices.extend(item.get("wiki_indices", []))
    unique_wiki_indices = sorted(set(all_wiki_indices))


    retrieved_paths: List[List[str]] = [
            faiss_id_to_chunk_leaf[str(i)]["path"] for i in unique_wiki_indices if str(i) in faiss_id_to_chunk_leaf
        ]
    retrieved_texts: List[str] = [
            faiss_id_to_chunk_leaf[str(i)]["text"] for i in unique_wiki_indices if str(i) in faiss_id_to_chunk_leaf
        ]

    if is_lc:
        retrieved_paths = [path for path in retrieved_paths if len(path) > 2]
        root_n = 1
    else:
        retrieved_paths = [path for path in retrieved_paths if len(path) > 1]
        root_n = 0

    root_paths = extract_root_paths(retrieved_paths, root_n=root_n)

    search_index_list, sum_list = process_all_subtrees_deep_parallel(
        root_paths=root_paths,
        query=query,
        wiki_ori_path=wiki_ori_path,
        sub_sum_chunks=sub_sum_chunks,
        segment_path=segment_path,
        model=model,
        dataset=dataset,
        wiki_prompt_json=wiki_prompt_json,
        total_max_concurrency=32,   
    )
    
    dedup_sum_list, sum_text_list = ids_to_summary_list(sum_list, sub_sum_chunks)
    wiki_all_lsit = search_index_list + dedup_sum_list
    
    search_global_ids = set(int(x) for x in search_index_list)
    all_seen_global_ids = set(int(x) for x in wiki_seen_ids)
    new_search_global_ids = search_global_ids - all_seen_global_ids
    updated_all_seen_global_ids = new_search_global_ids | all_seen_global_ids

    if is_lc:
        global2local_map_v = {v: k for k, v in local2global_map_v.items()}

        search_local_ids = {global2local_map_v[gid] for gid in new_search_global_ids if gid in global2local_map_v}
    else:
        search_local_ids = new_search_global_ids

    wiki_texts = [sub_chunk_map_v[str(i)]["text"] for i in search_local_ids]
    refs2 = "\n".join(wiki_texts)

    refs3 = "\n".join(sum_text_list)

    refs = refs1 + '\n' +refs2 + '\n' + refs3

    input_ids = embeder.tokenizer.encode(refs, add_special_tokens=False)
    if len(input_ids) > max_tokens:
        input_ids = input_ids[:max_tokens]   
        refs = embeder.tokenizer.decode(input_ids)
    num_tokens = len(input_ids)

    prediction = LLM_final(model=model, query=query, context=refs, dataset=dataset, wiki_prompt_json=wiki_prompt_json)

    if not wiki_all_lsit:
        has_wiki = False

    return prediction,updated_all_seen_global_ids , len(sum_list),has_wiki,num_tokens,count, refs



def navirag_note(query:str,
            title:str,
            model, 
            faiss_id_to_chunk_vanilla:dict,
            index_vanilla, 
            embeder,      
            is_lc:bool,      
            faiss_id_to_chunk_leaf:dict,

            top_k:int, 
            dataset:str=" ", 
            wiki_prompt_json=None,
            
            wiki_ori_path="",
            segment_path="",

            sum_chunks = None,
            max_tokens= 5120
            
)->str: 
    is_note = note_query_judge(query,model,dataset,wiki_prompt_json)

    if is_note:
        count = 0

        has_wiki = True

        seen_ids = set()
        
        if is_lc: 
            sub_index_v, local2global_map_v, sub_chunk_map_v = build_title_subindex(
                title=title,
                full_index=index_vanilla,
                chunk_map=faiss_id_to_chunk_vanilla
            )   
            cleaned_chunks = remove_empty_path_chunks(sum_chunks)

            sub_sum_chunks = filter_chunks_by_title(cleaned_chunks, title) 
        else:
            sub_index_v = index_vanilla
            sub_chunk_map_v = {str(i): v for i, v in faiss_id_to_chunk_vanilla.items()}
            sub_sum_chunks = remove_empty_path_chunks(sum_chunks)
            local2global_map_v = None

        query_vec = embeder.encode(
                query,
                convert_to_numpy=True,           
                normalize_embeddings=True      
                )

        query_vec = query_vec.reshape(1, -1) 

        new_ids, seen_ids = get_unique_faiss_ids(
            query_vec=query_vec,
            index=sub_index_v,
            seen_faiss_ids=seen_ids,
            base_topk=top_k
        )

        retrieved_wikis = [sub_chunk_map_v[str(i)] for i in new_ids]
        retrieved_texts_val = [item["text"] for item in retrieved_wikis]

        refs1 = "\n".join(retrieved_texts_val)

        if is_lc:
            global_ids = set(local2global_map_v[i] for i in seen_ids)
            wiki_seen_ids = [str(x) for x in global_ids]
            wiki_seen_ids_0 = wiki_seen_ids
        else:
            wiki_seen_ids = [str(x) for x in seen_ids]
            wiki_seen_ids_0 = wiki_seen_ids
            

        all_wiki_indices = [] 
        for item in retrieved_wikis:
            all_wiki_indices.extend(item.get("wiki_indices", []))
        unique_wiki_indices = sorted(set(all_wiki_indices))


        retrieved_paths: List[List[str]] = [
                faiss_id_to_chunk_leaf[str(i)]["path"] for i in unique_wiki_indices if str(i) in faiss_id_to_chunk_leaf
            ]
        retrieved_texts: List[str] = [
                faiss_id_to_chunk_leaf[str(i)]["text"] for i in unique_wiki_indices if str(i) in faiss_id_to_chunk_leaf
            ]

        if is_lc:
            retrieved_paths = [path for path in retrieved_paths if len(path) > 2]
            root_n = 1
        else:
            retrieved_paths = [path for path in retrieved_paths if len(path) > 1]
            root_n = 0


        root_paths = extract_root_paths(retrieved_paths, root_n=root_n)

        note = ''
        search_index_list = []
        sum_list = []
        leaf_text_list = []
        

        for root_list in root_paths:

            full_sub_tree = load_sub_tree_from_path(root_list, wiki_file_path=wiki_ori_path)

            root = root_list[-1]

            subtree_sum_chunks = filter_and_trim_paths(sub_sum_chunks,root_list)


            search_index_list,wiki_seen_ids,sum_list, leaf_text_list,count,note = read_search_note_step(root,None,wiki_seen_ids,search_index_list,sum_list,leaf_text_list,count,note,is_lc,local2global_map_v,sub_chunk_map_v, query, full_sub_tree,segment_path,model,dataset,subtree_sum_chunks,wiki_prompt_json,embeder.tokenizer)
        
        wiki_all_lsit = search_index_list + sum_list
            
        all_seen_global_ids = set(int(x) for x in wiki_seen_ids)

        search_global_ids = set(int(x) for x in search_index_list)
        if is_lc:

            global2local_map_v = {v: k for k, v in local2global_map_v.items()}


            search_local_ids = {global2local_map_v[gid] for gid in search_global_ids if gid in global2local_map_v}
        else:
            search_local_ids = search_global_ids

        wiki_texts = [sub_chunk_map_v[str(i)]["text"] for i in search_local_ids]

        contetx1 = "\n".join(retrieved_texts_val)
        context2 = "\n".join(wiki_texts)
        context3 = "\n".join(sum_list)
        context_final = contetx1 + context2 + context3
        if not wiki_all_lsit:
            has_wiki = False
            note = contetx1
        else:
            note = note_refine(note,retrieved_texts_val,query,model,dataset,wiki_prompt_json,embeder.tokenizer)


        num_tokens = len(embeder.tokenizer.tokenize(note))
        prediction = LLM_final(model=model, query=query, context=note, dataset=dataset, wiki_prompt_json=wiki_prompt_json)

        
        return prediction,all_seen_global_ids , len(sum_list),has_wiki,num_tokens,count,context_final, note, is_note
    else:

        prediction,all_seen_global_ids,sum_len,has_wiki,num_tokens,count,refs = navirag( 
                query=query,
                title=title,
                model=model, 
                faiss_id_to_chunk_vanilla=faiss_id_to_chunk_vanilla,
                index_vanilla=index_vanilla, 
                faiss_id_to_chunk_leaf=faiss_id_to_chunk_leaf,
                embeder=embeder,      
                is_lc=is_lc,      
            
                top_k=top_k, 
                dataset=dataset, 
                wiki_prompt_json=wiki_prompt_json,

                wiki_ori_path =wiki_ori_path,
                segment_path = segment_path,

                sum_chunks = sum_chunks,
                max_tokens= max_tokens
                )

        return prediction,all_seen_global_ids , sum_len, has_wiki,num_tokens,count, refs,refs,is_note

