from __future__ import annotations
import logging
import os
import re
import time
from typing import Optional, List
from urllib import request, error
import json
import argparse
from dotenv import load_dotenv
from mcp import Tool
from mcp.server.fastmcp import FastMCP
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mysql.connector import Error, connect
from bs4 import BeautifulSoup
import certifi
import ssl
from pydantic import BaseModel
from pyobvector import ObVecClient, MatchAgainst, l2_distance, inner_product, cosine_distance
from sqlalchemy import text

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("dingodb_mcp_server")

load_dotenv()

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
EMBEDDING_MODEL_PROVIDER = os.getenv("EMBEDDING_MODEL_PROVIDER", "huggingface")
ENABLE_MEMORY = int(os.getenv("ENABLE_MEMORY", 0))

TABLE_NAME_MEMORY = os.getenv("TABLE_NAME_MEMORY", "dingo_mcp_memory")

logger.info(
    f" ENABLE_MEMORY: {ENABLE_MEMORY},EMBEDDING_MODEL_NAME: {EMBEDDING_MODEL_NAME}, EMBEDDING_MODEL_PROVIDER: {EMBEDDING_MODEL_PROVIDER}"
)


class DingoConnection(BaseModel):
    host: str
    port: int
    user: str
    password: str
    database: str


class DingoMemoryItem(BaseModel):
    mem_id: int = None
    content: str
    meta: dict
    embedding: List[float]


# Check if authentication should be enabled based on ALLOWED_TOKENS
# This check happens after load_dotenv() so it can read from .env file
allowed_tokens_str = os.getenv("ALLOWED_TOKENS", "")
enable_auth = bool(allowed_tokens_str.strip())


class SimpleTokenVerifier(TokenVerifier):
    """
    Simple token verifier that validates tokens against a list of allowed tokens.
    Configure allowed tokens via ALLOWED_TOKENS environment variable (comma-separated).
    """

    def __init__(self):
        # Get allowed tokens from environment variable
        allowed_tokens_str = os.getenv("ALLOWED_TOKENS", "")
        self.allowed_tokens = set(
            token.strip() for token in allowed_tokens_str.split(",") if token.strip()
        )

        logger.info(f"Token verifier initialized with {len(self.allowed_tokens)} allowed tokens")

    async def verify_token(self, token: str) -> AccessToken | None:
        """
        Verify a bearer token.

        Args:
            token: The token to verify

        Returns:
            AccessToken if valid, None if invalid
        """
        # Check if token is empty
        if not token or not token.strip():
            logger.debug("Empty token provided")
            return None

        # Check if token is in allowed list
        if token not in self.allowed_tokens:
            logger.warning(f"Invalid token provided: {token[:10]}...")
            return None

        logger.debug(f"Valid token accepted: {token[:10]}...")
        return AccessToken(
            token=token, client_id="authenticated_client", scopes=["read", "write"], expires_at=None
        )


db_conn_info = DingoConnection(
    # host=os.getenv("DINGODB_HOST", "10.230.109.45"),
    # port=os.getenv("DINGODB_PORT", 3306),
    # user=os.getenv("DINGODB_USER", "root"),
    # password=os.getenv("DINGODB_PASSWORD",""),
    # database=os.getenv("DINGODB_DATABASE","dingo"),

    host=os.getenv("DINGODB_HOST", "172.30.14.123"),
    port=os.getenv("DINGODB_PORT", 3307),
    user=os.getenv("DINGODB_USER", "root"),
    password=os.getenv("DINGODB_PASSWORD", ""),
    database=os.getenv("DINGODB_DATABASE", "dingo"),
)

if enable_auth:
    logger.info("Authentication enabled - ALLOWED_TOKENS configured")
    # Initialize server with token verifier and minimal auth settings
    # FastMCP requires auth settings when using token_verifier
    app = FastMCP(
        "dingodb_mcp_server",
        token_verifier=SimpleTokenVerifier(),
        auth=AuthSettings(
            # Because the TokenVerifier is being used, the following two URLs only need to comply with the URL rules.
            issuer_url="http://localhost:8000",
            resource_server_url="http://localhost:8000",
        ),
    )
else:
    # Initialize server without authentication
    app = FastMCP("dingodb_mcp_server")


@app.resource("dingo://sample/{table}", description="table sample")
def table_sample(table: str) -> str:
    valid_table_pattern = re.compile(r'^[a-zA-Z0-9_.]+$')
    if not valid_table_pattern.match(table):
        logger.error(f"Invalid table name: {table} (contains illegal characters)")
        return f"Failed to sample table: invalid table name '{table}'"
    try:
        with connect(**db_conn_info.model_dump()) as conn:
            with conn.cursor() as cursor:
                query = f"SELECT * FROM `{table}` LIMIT 100"
                cursor.execute(query)
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                result = [",".join(map(str, row)) for row in rows]
                return "\n".join([",".join(columns)] + result)

    except Error as e:
        logger.error(f"Failed to list tables: {str(e)}")
        return f"Failed to sample table: {table}"


@app.resource("dingo://tables", description="list all tables")
def list_tables() -> str:
    """List Dingodb tables as resources."""
    try:
        with connect(**db_conn_info.model_dump()) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SHOW TABLES")
                tables = cursor.fetchall()
                resp_header = "Tables of this table: \n"
                columns = [desc[0] for desc in cursor.description]
                result = [",".join(map(str, table)) for table in tables]
                return resp_header + ("\n".join([",".join(columns)] + result))
    except Error as e:
        logger.error(f"Failed to list tables: {str(e)}")
        return "Failed to list tables"

@app.tool()
def execute_sql(sql: str) -> str:
    """Execute an SQL on the Dingodb server."""
    logger.info(f"Calling tool: execute_sql  with arguments: {sql}")
    result = {"sql": sql, "success": False, "rows": 0, "columns": None, "data": None, "error": None}
    try:
        with connect(**db_conn_info.model_dump()) as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                if cursor.description:
                    result["columns"] = [column[0] for column in cursor.description]
                    result["data"] = [[str(cell) for cell in row] for row in cursor.fetchall()]
                else:
                    conn.commit()
                result["rows"] = cursor.rowcount
                result["success"] = True
    except Error as e:
        # result["error"] = f"[Error]: {e}"
        # 提取错误码和描述（适配 Dingodb/MySQL 异常格式）
        error_msg = str(e)
        result["error"] = f"[Error]: {error_msg}"
        # 尝试提取错误码（比如 1064）
        if "(" in error_msg and ")" in error_msg:
            result["error_code"] = error_msg.split("(")[0].strip()
    except Exception as e:
        result["error"] = f"[Exception]: {e}"
    json_result = json.dumps(result)
    if result["error"]:
        logger.error(f"SQL executed failed, result: {json_result}")
    return json_result


@app.tool()
def get_ob_ash_report(
    start_time: str,
    end_time: str,
    tenant_id: Optional[int] = None,
) -> str:
    """
    Get Dingodb Active Session History report.
    ASH can sample the status of all Active Sessions in the system at 1-second intervals, including:
        Current executing SQL ID
        Current wait events (if any)
        Wait time and wait parameters
        The module where the SESSION is located during sampling (PARSE, EXECUTE, PL, etc.)
        SESSION status records, such as SESSION MODULE, ACTION, CLIENT ID
    This will be very useful when you perform performance analysis.RetryClaude can make mistakes. Please double-check responses.

    Args:
        start_time: Sample Start Time,Format: yyyy-MM-dd HH:mm:ss.
        end_time: Sample End Time,Format: yyyy-MM-dd HH:mm:ss.
        tenant_id: Used to specify the tenant ID for generating the ASH Report. Leaving this field blank or setting it to NULL indicates no restriction on the TENANT_ID.
    """
    logger.info(
        f"Calling tool: get_ob_ash_report  with arguments: {start_time}, {end_time}, {tenant_id}"
    )
    # Construct the SQL query
    sql_query = f"""
        CALL DBMS_WORKLOAD_REPOSITORY.ASH_REPORT('{start_time}','{end_time}', NULL, NULL, NULL, 'TEXT', NULL, NULL, {tenant_id if tenant_id is not None else "NULL"});
    """
    try:
        return execute_sql(sql_query)
    except Error as e:
        logger.error(f"Error get ASH report,executing SQL '{sql_query}': {e}")
        return f"Error get ASH report,{str(e)}"


@app.tool(name="get_current_time", description="Get current time")
def get_current_time() -> str:
    """Get current time from Dingodb database."""
    logger.info("Calling tool: get_current_time")
    sql_query = "SELECT NOW()"
    try:
        return execute_sql(sql_query)
    except Error as e:
        logger.error(f"Error getting database time: {e}")
        # Fallback to system time if database query fails
        local_time = time.localtime()
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
        logger.info(f"Fallback to system time: {formatted_time}")
        return formatted_time


@app.tool()
def get_current_tenant() -> str:
    """
    Get the current tenant name from dingo.
    """
    logger.info("Calling tool: get_current_tenant")
    sql_query = "SELECT TENANT_NAME,TENANT_ID FROM dingo.DBA_OB_TENANTS"
    try:
        return execute_sql(sql_query)
    except Error as e:
        logger.error(f"Error executing SQL '{sql_query}': {e}")
        return f"Error executing query: {str(e)}"


@app.tool()
def get_all_server_nodes():
    """
    Get all server nodes from dingo.
    You need to be sys tenant to get all server nodes.
    """
    tenant = json.loads(get_current_tenant())["data"][0][0]
    if tenant != "sys":
        raise ValueError("Only sys tenant can get all server nodes")

    logger.info("Calling tool: get_all_server_nodes")
    sql_query = "select * from dingo.DBA_OB_SERVERS"
    try:
        return execute_sql(sql_query)
    except Error as e:
        logger.error(f"Error executing SQL '{sql_query}': {e}")
        return f"Error executing query: {str(e)}"


@app.tool()
def get_resource_capacity():
    """
    Get resource capacity from dingo.
    You need to be sys tenant to get resource capacity.
    """
    tenant = json.loads(get_current_tenant())["data"][0][0]
    if tenant != "sys":
        raise ValueError("Only sys tenant can get resource capacity")
    logger.info("Calling tool: get_resource_capacity")
    sql_query = "select * from dingo.GV$OB_SERVERS"
    try:
        return execute_sql(sql_query)
    except Error as e:
        logger.error(f"Error executing SQL '{sql_query}': {e}")
        return f"Error executing query: {str(e)}"


@app.tool()
def search_oceanbase_document(keyword: str) -> str:
    """
    This tool is designed to provide context-specific information about Dingodb to a large language model (LLM) to enhance the accuracy and relevance of its responses.
    The LLM should automatically extracts relevant search keywords from user queries or LLM's answer for the tool parameter "keyword".
    The main functions of this tool include:
    1.Information Retrieval: The MCP Tool searches through Dingodb-related documentation using the extracted keywords, locating and extracting the most relevant information.
    2.Context Provision: The retrieved information from Dingodb documentation is then fed back to the LLM as contextual reference material. This context is not directly shown to the user but is used to refine and inform the LLM’s responses.
    This tool ensures that when the LLM’s internal documentation is insufficient to generate high-quality responses, it dynamically retrieves necessary Dingodb information, thereby maintaining a high level of response accuracy and expertise.
    Important: keyword must be Chinese
    """
    logger.info(f"Calling tool: search_oceanbase_document,keyword:{keyword}")
    search_api_url = (
        "https://cn-wan-api.oceanbase.com/wanApi/forum/docCenter/productDocFile/v3/searchDocList"
    )
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "Origin": "https://www.oceanbase.com",
        "Referer": "https://www.oceanbase.com/",
    }
    qeury_param = {
        "pageNo": 1,
        "pageSize": 5,  # Search for 5 results at a time.
        "query": keyword,
    }
    # Turn the dictionary into a JSON string, then change it to bytes
    qeury_param = json.dumps(qeury_param).encode("utf-8")
    req = request.Request(search_api_url, data=qeury_param, headers=headers, method="POST")
    # Create an SSL context using certifi to fix HTTPS errors.
    context = ssl.create_default_context(cafile=certifi.where())
    try:
        with request.urlopen(req, timeout=5, context=context) as response:
            response_body = response.read().decode("utf-8")
            json_data = json.loads(response_body)
            # In the results, we mainly need the content in the data field.
            data_array = json_data["data"]  # Parse JSON response
            result_list = []
            for item in data_array:
                doc_url = "https://www.dingo.com/docs/" + item["urlCode"] + "-" + item["id"]
                logger.info(f"doc_url:${doc_url}")
                content = get_ob_doc_content(doc_url, item["id"])
                result_list.append(content)
            return json.dumps(result_list, ensure_ascii=False)
    except error.HTTPError as e:
        logger.error(f"HTTP Error: {e.code} - {e.reason}")
        return "No results were found"
    except error.URLError as e:
        logger.error(f"URL Error: {e.reason}")
        return "No results were found"


def get_ob_doc_content(doc_url: str, doc_id: str) -> dict:
    doc_param = {"id": doc_id, "url": doc_url}
    doc_param = json.dumps(doc_param).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "Origin": "https://www.dingo.com",
        "Referer": "https://www.dingo.com/",
    }
    doc_api_url = (
        "https://cn-wan-api.dingo.com/wanApi/forum/docCenter/productDocFile/v4/docDetails"
    )
    req = request.Request(doc_api_url, data=doc_param, headers=headers, method="POST")
    # Make an SSL context with certifi to fix HTTPS errors.
    context = ssl.create_default_context(cafile=certifi.where())
    try:
        with request.urlopen(req, timeout=5, context=context) as response:
            response_body = response.read().decode("utf-8")
            json_data = json.loads(response_body)
            # In the results, we mainly need the content in the data field.
            data = json_data["data"]
            # The docContent field has HTML text.
            soup = BeautifulSoup(data["docContent"], "html.parser")
            # Remove script, style, nav, header, and footer elements.
            for element in soup(["script", "style", "nav", "header", "footer"]):
                element.decompose()
            # Remove HTML tags and keep only the text.
            text = soup.get_text()
            # Remove spaces at the beginning and end of each line.
            lines = (line.strip() for line in text.splitlines())
            # Remove empty lines.
            text = "\n".join(line for line in lines if line)
            logger.info(f"text length:{len(text)}")
            # If the text is too long, only keep the first 8000 characters.
            if len(text) > 8000:
                text = text[:8000] + "... [content truncated]"
            # Reorganize the final result. The tdkInfo field should include the document's title, description, and keywords.
            tdkInfo = data["tdkInfo"]
            final_result = {
                "title": tdkInfo["title"],
                "description": tdkInfo["description"],
                "keyword": tdkInfo["keyword"],
                "content": text,
                "oceanbase_version": data["version"],
                "content_updatetime": data["docGmtModified"],
            }
            return final_result
    except error.HTTPError as e:
        logger.error(f"HTTP Error: {e.code} - {e.reason}")
        return {"result": "No results were found"}
    except error.URLError as e:
        logger.error(f"URL Error: {e.reason}")
        return {"result": "No results were found"}


@app.tool(
    # name="dingodb_text_search",
    # description="DingoDB全文检索工具，支持指定索引名检索表中指定列包含关键词的数据",
   )
def dingodb_text_search(
    table_name: str,
    index_name: str,
    full_text_search_column_name: str,
    full_text_search_expr: str,
    condition_query: Optional[list[str]] = None,
    output_column_name: Optional[list[str]] = None,
    limit: int = 5
) -> str:
    """
    Search for documents using full text search in a Dingodb table.

    Args:
        table_name: Name of the table to search.
        index_name: Specifying the query index.
        full_text_search_column_name: Specify the column to be searched in full text.
        full_text_search_expr: Specify the keywords or phrases to search for.
        condition_query: Other WHERE condition query statements except full-text search.
        limit: Maximum number of results to return.
        # output_column_name: columns to include in results.
    Returns:
        JSON格式的检索结果
    """
    # logger.info(
    #     f"Calling tool: dingodb_text_search  with arguments: {table_name}, {full_text_search_column_name}, {full_text_search_expr}"
    # )
    if not table_name:
        error_result = {
            "sql": "",
            "success": False,
            "rows": 0,
            "columns": None,
            "data": None,
            "error": "[Error]: 表名table_name为必填参数"
        }
        return json.dumps(error_result)
    if not full_text_search_column_name or not full_text_search_expr:
        error_result = {
            "sql": "",
            "success": False,
            "rows": 0,
            "columns": None,
            "data": None,
            "error": "[Error]: 检索列列表full_text_search_column_name和检索表达式full_text_search_expr为必填"
        }
        return json.dumps(error_result)
    if not index_name:
        return json.dumps({
            "sql": "", "success": False,
            "error": "[Error]: 索引名index_name为必填参数"
        }, ensure_ascii=False)
    if output_column_name and isinstance(output_column_name, list) and len(output_column_name) > 0:
        output_cols = ", ".join([f"`{col.strip()}`" for col in output_column_name])
    else:
        output_cols = "*"

    # 3. 拼接全文检索WHERE条件
    # 转义检索表达式中的单引号，防SQL注入
    escaped_expr = full_text_search_expr.replace("'", "''")
    # 拆分检索表达式为关键词（按空格拆分）
    keywords = [kw.strip() for kw in escaped_expr.split() if kw.strip()]
    if not keywords:
        error_result = {
            "sql": "",
            "success": False,
            "rows": 0,
            "columns": None,
            "data": None,
            "error": "[Error]: 检索表达式full_text_search_expr不能为空或仅含空格"
        }
        return json.dumps(error_result)

    # 为每个检索列拼接关键词匹配条件（多列+多关键词：列1 LIKE %kw1% AND 列1 LIKE %kw2% AND 列2 LIKE %kw1% ...）
    text_search_conditions = []
    col_escaped = f"{full_text_search_column_name.strip()}"
    for kw in keywords:
        text_search_conditions.append(f"{col_escaped}:{kw}")
    text_search_clause = " AND ".join(text_search_conditions)

    where_conditions = []
    if condition_query and isinstance(condition_query, list) and len(condition_query) > 0:
        # 校验并转义额外条件（仅做基础格式校验，需用户保证条件合法性）
        for clause in condition_query:
            if clause.strip():
                where_conditions.append(clause.strip())
    where_clause = " AND ".join(where_conditions)

    sql = (f"SELECT {output_cols} FROM text_search({table_name.strip()},"
           f" {index_name.strip()}, '{text_search_clause}',{int(limit)})")
    if len(where_clause) > 0:
        sql += " WHERE " + where_clause
    logger.info(f"Calling tool: dingodb_text_search with generated SQL: {sql}")
    return execute_sql(sql)


@app.tool()
def dingodb_vector_search(
    table_name: str,
    vector_data: list[float],
    vec_column_name: str = "vector",
    # distance_func: Optional[str] = "l2",
    # with_distance: Optional[bool] = True,
    topk: int = 5,
    output_column_name: Optional[list[str]] = None,
) -> str:
    """
    Perform vector similarity search on a Dingodb table.

    Args:
        table_name: Name of the table to search.
        vector_data: Query vector.
        vec_column_name: column name containing vectors to search.
        # distance_func: The index distance algorithm used when comparing the distance between two vectors.
        # with_distance: Whether to output distance data.
        topk: Number of results returned.
        output_column_name: Returned table fields.
    """
    logger.info(
        f"Calling tool: oceabase_vector_search  with arguments: {table_name}, {vector_data[:10]}, {vec_column_name}"
    )
    if not table_name:
        error_result = {
            "sql": "",
            "success": False,
            "rows": 0,
            "columns": None,
            "data": None,
            "error": "[Error]: 表名table_name为必填参数"
        }
        return json.dumps(error_result)
    if not isinstance(vector_data, list) or len(vector_data) == 0:
        return json.dumps({
            "sql": "", "success": False, "rows": 0, "columns": None, "data": None,
            "error": "[Error]: 向量数据vector_data必须是非空浮点型列表"
        }, ensure_ascii=False)
    try:
        vector_data = [float(vec) for vec in vector_data]
    except ValueError:
        return json.dumps({
            "sql": "", "success": False, "rows": 0, "columns": None, "data": None,
            "error": "[Error]: 向量数据vector_data必须全为浮点型数值"
        }, ensure_ascii=False)
    if output_column_name and isinstance(output_column_name, list) and len(output_column_name) > 0:
        output_cols = ", ".join([f"`{col.strip()}`" for col in output_column_name])
    else:
        output_cols = "*"
    # 向量格式转换：列表→OB支持的向量字符串（如"[0.1,0.2,0.3]"）
    vector_str = json.dumps(vector_data).replace(" ", "")  # 去除空格，适配OB语法
    # 向量列转义
    vec_col_escaped = f"`{vec_column_name.strip()}`"
    # 表名转义
    table_escaped = f"`{table_name.strip()}`"

    # Dingodb向量检索核心语法（余弦相似度降序，取topk）
    # 注：OB向量函数可根据实际版本调整（如cosine_distance/l2_distance）
    sql = f"""SELECT {output_cols} FROM vector({table_escaped}, {vec_col_escaped}, array{vector_str},{topk})""".strip()  # ASC：余弦相似度值越小，相似度越高
    logger.info(f"Calling tool: Dingodb_vector_search with vector: {vector_str}, topk: {topk}")
    return execute_sql(sql)


@app.tool()
def dingodb_hybrid_search(
    table_name: str,
    vector_data: list[float],
    vec_column_name: str = "vector",
    # distance_func: Optional[str] = "l2",
    # with_distance: Optional[bool] = True,
    filter_expr: Optional[list[str]] = None,
    topk: int = 5,
    output_column_name: Optional[list[str]] = None,
) -> str:
    """
    Perform hybrid search combining relational condition filtering(that is, scalar) and vector search.

    Args:
        table_name: Name of the table to search.
        vector_data: Query vector.
        vec_column_name: column name containing vectors to search.
        # distance_func: The index distance algorithm used when comparing the distance between two vectors.
        # with_distance: Whether to output distance data.
        filter_expr: Scalar conditions requiring filtering in where clause.
        topk: Number of results returned.
        output_column_name: Returned table fields,unless explicitly requested, please do not provide.
    """
    logger.info(
        f"""Calling tool: dingodb_hybrid_search  with arguments: {table_name}, {vector_data[:10]}, {vec_column_name}
        ,{filter_expr}"""
    )
    # 1. 基础参数校验
    if not table_name:
        return json.dumps({
            "sql": "", "success": False, "rows": 0, "columns": None, "data": None,
            "error": "[Error]: 表名table_name为必填参数"
        }, ensure_ascii=False)

    if not isinstance(vector_data, list) or len(vector_data) == 0:
        return json.dumps({
            "sql": "", "success": False, "rows": 0, "columns": None, "data": None,
            "error": "[Error]: 向量数据vector_data必须是非空浮点型列表"
        }, ensure_ascii=False)

    # 校验向量元素类型（确保是浮点型）
    try:
        vector_data = [float(vec) for vec in vector_data]
    except ValueError:
        return json.dumps({
            "sql": "", "success": False, "rows": 0, "columns": None, "data": None,
            "error": "[Error]: 向量数据vector_data必须全为浮点型数值"
        }, ensure_ascii=False)

    # 2. 处理输出列
    if output_column_name and isinstance(output_column_name, list) and len(output_column_name) > 0:
        output_cols = ", ".join([f"`{col.strip()}`" for col in output_column_name])
    else:
        output_cols = "*"

    # 3. 拼接DingoDB混合检索SQL（向量检索+条件过滤）
    # 向量格式转换：列表→DingoDB支持的向量字符串
    vector_str = json.dumps(vector_data).replace(" ", "")  # 去除空格，适配语法
    # 向量列/表名转义
    vec_col_escaped = f"`{vec_column_name.strip()}`"
    table_escaped = f"`{table_name.strip()}`"

    # 基础向量检索SQL（使用余弦距离排序）
    sql_base = f"""SELECT {output_cols} FROM vector({table_escaped}, {vec_col_escaped}, array{vector_str},{topk})""".strip()  # ASC：余弦相似度值越小，相似度越高


    # 拼接过滤条件（filter_expr）
    where_clause_list = []
    if filter_expr and isinstance(filter_expr, list) and len(filter_expr) > 0:
        for expr in filter_expr:
            if expr.strip():  # 过滤空条件
                where_clause_list.append(expr.strip())

    # 组合完整SQL
    if where_clause_list:
        sql = f"{sql_base} WHERE {' AND '.join(where_clause_list)} "
    else:
        sql = f"{sql_base} "

    # 4. 调用execute_sql执行混合检索
    logger.info(f"Calling tool: dingodb_hybrid_search with filter: {filter_expr}, topk: {topk}")
    return execute_sql(sql)

def main():
    """Main entry point to run the MCP server."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transport",
        type=str,
        default="stdio",
        help="Specify the MCP server transport type as stdio or sse or streamable-http.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=11000, help="Port to listen on")
    args = parser.parse_args()
    transport = args.transport
    logger.info(f"Starting Dingodb MCP server with {transport} mode...")
    if transport in {"sse", "streamable-http"}:
        app.settings.host = args.host
        app.settings.port = args.port
    app.run(transport=transport)


if __name__ == "__main__":
    main()
