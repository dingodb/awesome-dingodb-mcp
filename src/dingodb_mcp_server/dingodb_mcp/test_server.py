from .server import execute_sql,list_tables,table_sample,get_current_time,search_oceanbase_document,dingodb_text_search,dingodb_vector_search,dingodb_hybrid_search

def test_execute_sql():
    res = execute_sql("SELECT * FROM `dingo`.`dingospeed`")
    print(res)

def test_list_tables():
    res = list_tables()
    print(res)

def test_table_sample():
    res = table_sample("cache_job")
    print(res)

def test_get_current_time():
    res = get_current_time()
    print(res)


def test_search_oceanbase_document():
    res = search_oceanbase_document("1")
    print(res)

def test_dingodb_text_search():
    res = dingodb_text_search("t2", "text_index",["description"], "Ergonomic")
    # res = dingodb_text_search("t2", ["description"], "keyboard", "text_index",["text_id >=4"])
    print(res)

def test_oceabase_vector_search():
    res = dingodb_vector_search("t2",[0.8894774317741394, 0.7277960181236267, 0.692345142364502, 0.47235092520713806, 0.8568729162216187, 0.6647433042526245, 0.3333759307861328, 0.5181455016136169],"feature")
    print(res)

def test_dingodb_hybrid_search():
    res = dingodb_hybrid_search("t2",[0.8894774317741394, 0.7277960181236267, 0.692345142364502, 0.47235092520713806, 0.8568729162216187, 0.6647433042526245, 0.3333759307861328, 0.5181455016136169],"feature", ["rating = 5"])
    print(res)