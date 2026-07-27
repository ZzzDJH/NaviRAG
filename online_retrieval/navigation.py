from utiles import *


async def leaf_select_parallel_async(
    text,
    query,
    model,
    dataset,
    wiki_prompt_json,
    sem,
):

    async with sem:
        return await asyncio.to_thread(
            leaf_select_parallel,
            text,
            query,
            [],   
            model,
            dataset,
            wiki_prompt_json,
        )


async def read_select_titles_parallel_async(
    root,
    path,
    titles,
    query,
    model,
    dataset,
    subtree_sum_chunks,
    wiki_prompt_json,
    sem,
):

    async with sem:
        return await asyncio.to_thread(
            read_select_titles_parallel,
            root,
            path,
            titles,
            [],   
            query,
            model,
            dataset,
            subtree_sum_chunks,
            wiki_prompt_json,
        )




async def process_title_branch_async(
    root,
    path,
    title,
    current_dict,
    query,
    current_wiki,
    segment_path,
    model,
    dataset,
    subtree_sum_chunks,
    wiki_prompt_json,
    sem,
):

    value = current_dict[title]

    if isinstance(value, str):
        text = value
        local_search_index_list = await leaf_select_parallel_async(
            text=text,
            query=query,
            model=model,
            dataset=dataset,
            wiki_prompt_json=wiki_prompt_json,
            sem=sem,
        )
        return local_search_index_list, []

    elif isinstance(value, dict):
        return await read_search_parallel_async(
            root=root,
            path=path + [title],
            query=query,
            current_wiki=current_wiki,
            segment_path=segment_path,
            model=model,
            dataset=dataset,
            subtree_sum_chunks=subtree_sum_chunks,
            wiki_prompt_json=wiki_prompt_json,
            sem=sem,
        )

    return [], []




async def read_search_parallel_async(
    root,                  
    path,                  
    query,
    current_wiki,
    segment_path,
    model,
    dataset,
    subtree_sum_chunks,
    wiki_prompt_json,
    sem,                   
):
    
    if path is None:
        path = []

    current_dict = get_by_path(current_wiki, path)

    if isinstance(current_dict, str):
        text = current_dict
        local_search_index_list = await leaf_select_parallel_async(
            text=text,
            query=query,
            model=model,
            dataset=dataset,
            wiki_prompt_json=wiki_prompt_json,
            sem=sem,
        )
        return local_search_index_list, []

    titles = list(current_dict.keys())

    selected_titles, current_sum_list = await read_select_titles_parallel_async(
        root=root,
        path=path,
        titles=titles,
        query=query,
        model=model,
        dataset=dataset,
        subtree_sum_chunks=subtree_sum_chunks,
        wiki_prompt_json=wiki_prompt_json,
        sem=sem,
    )

    if selected_titles is None:
        return [], current_sum_list

    tasks = [
        process_title_branch_async(
            root=root,
            path=path,
            title=title,
            current_dict=current_dict,
            query=query,
            current_wiki=current_wiki,
            segment_path=segment_path,
            model=model,
            dataset=dataset,
            subtree_sum_chunks=subtree_sum_chunks,
            wiki_prompt_json=wiki_prompt_json,
            sem=sem,
        )
        for title in selected_titles
    ]

    branch_results = await asyncio.gather(*tasks, return_exceptions=True)

  
    merged_search_index_list = []
    merged_sum_list = list(current_sum_list) if current_sum_list is not None else []

    for result in branch_results:
        if isinstance(result, Exception):
            print(f"[read_search_parallel_async] branch failed: {result}")
            continue

        branch_search_index_list, branch_sum_list = result
        merged_search_index_list.extend(branch_search_index_list)
        merged_sum_list.extend(branch_sum_list)

    return merged_search_index_list, merged_sum_list



async def process_one_subtree_async(
    root_list,
    query,
    wiki_ori_path,
    sub_sum_chunks,
    segment_path,
    model,
    dataset,
    wiki_prompt_json,
    sem,
):

    full_sub_tree = load_sub_tree_from_path(
        root_list,
        wiki_file_path=wiki_ori_path
    )

    root = root_list[-1]
    subtree_sum_chunks = filter_and_trim_paths(sub_sum_chunks, root_list)

    local_search_index_list, local_sum_list = await read_search_parallel_async(
        root=root,
        path=None,
        query=query,
        current_wiki=full_sub_tree,
        segment_path=segment_path,
        model=model,
        dataset=dataset,
        subtree_sum_chunks=subtree_sum_chunks,
        wiki_prompt_json=wiki_prompt_json,
        sem=sem,
    )

    return local_search_index_list, local_sum_list


async def process_all_subtrees_async(
    root_paths,
    query,
    wiki_ori_path,
    sub_sum_chunks,
    segment_path,
    model,
    dataset,
    wiki_prompt_json,
    total_max_concurrency=8,   
):
    
    sem = asyncio.Semaphore(total_max_concurrency)

    tasks = [
        process_one_subtree_async(
            root_list=root_list,
            query=query,
            wiki_ori_path=wiki_ori_path,
            sub_sum_chunks=sub_sum_chunks,
            segment_path=segment_path,
            model=model,
            dataset=dataset,
            wiki_prompt_json=wiki_prompt_json,
            sem=sem,
        )
        for root_list in root_paths
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    search_index_list = []
    sum_list = []

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"[process_all_subtrees_async] subtree {i} failed: {result}")
            continue

        local_search_index_list, local_sum_list = result
        search_index_list.extend(local_search_index_list)
        sum_list.extend(local_sum_list)

    return search_index_list, sum_list



def process_all_subtrees_deep_parallel(
    root_paths,
    query,
    wiki_ori_path,
    sub_sum_chunks,
    segment_path,
    model,
    dataset,
    wiki_prompt_json,
    total_max_concurrency=8,
):

    return asyncio.run(
        process_all_subtrees_async(
            root_paths=root_paths,
            query=query,
            wiki_ori_path=wiki_ori_path,
            sub_sum_chunks=sub_sum_chunks,
            segment_path=segment_path,
            model=model,
            dataset=dataset,
            wiki_prompt_json=wiki_prompt_json,
            total_max_concurrency=total_max_concurrency,
        )
    )




def read_search_note_step(root, 
                path, 
                wiki_seen_ids,
                search_index_list,
                sum_list,
                leaf_text_list,
                count,
                note,
                is_lc,
                local2global_map_v,
                sub_chunk_map_v,
                query, 
                current_wiki,
                segment_path,
                model,
                dataset,
                subtree_sum_chunks,
                wiki_prompt_json,
                tokenizer
                
                ):
    if path is None:
        path = []

    current_dict = get_by_path(current_wiki, path)

    

    if isinstance(current_dict, str) :
        text = current_dict
        search_index_list,wiki_seen_ids,leaf_text_list,note = leaf_select_note_step(text,query, search_index_list,wiki_seen_ids,leaf_text_list, note, is_lc,local2global_map_v,sub_chunk_map_v,model=model, dataset=dataset,wiki_prompt_json=wiki_prompt_json,tokenizer=tokenizer )

        return search_index_list,wiki_seen_ids, sum_list,leaf_text_list,count,note

    titles = list(current_dict.keys())

    selected_titles,sum_list,note = read_select_titles_with_note(root,path, titles,sum_list,query,note, model=model, dataset=dataset,subtree_sum_chunks=subtree_sum_chunks,wiki_prompt_json=wiki_prompt_json,tokenizer=tokenizer)

    count += 1

    
    if selected_titles is None:
        return search_index_list,wiki_seen_ids, sum_list,leaf_text_list,count,note
    

    for title in selected_titles:
        value = current_dict[title]

        if isinstance(value, str):
            text = value
            search_index_list,wiki_seen_ids,leaf_text_list,note = leaf_select_note_step(text,query, search_index_list,wiki_seen_ids,leaf_text_list, note, is_lc,local2global_map_v,sub_chunk_map_v,model=model, dataset=dataset,wiki_prompt_json=wiki_prompt_json,tokenizer=tokenizer )

            continue

        elif isinstance(value, dict):
            
            search_index_list,wiki_seen_ids, sum_list,leaf_text_list,count,note = read_search_note_step(root,path + [title],wiki_seen_ids,search_index_list, sum_list,leaf_text_list,count,note,is_lc,local2global_map_v,sub_chunk_map_v,query,current_wiki,segment_path,model,dataset,subtree_sum_chunks,wiki_prompt_json,tokenizer)

    return search_index_list,wiki_seen_ids, sum_list,leaf_text_list,count,note