import pyodbc as odbc
import sys

server     = 'localhost'
database   = 'Quantum_MaxCut_20240808'
#username   = ''
#password   = ''
#connString = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID=' + username + ';PWD=' + password
connString = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + server + ';DATABASE=' + database + ';Trusted_Connection=yes';

def insert_db(table, pk_name, table_fields, record):
    """
    For a selected table, insert new records in our database

    Args:
        table        : str   | the name of the table
        pk_name      : str   | the name of the primary key field
        table_fields : str[] | array of fields that will be inserted in the insert command
        record       : array | (columns/fields) (or tuple) with content to be inserted

    Returns:
        pk           : int   | new primary key of the row (note: pk = result.fetchone()[0])
    """
    try:
        conn = odbc.connect(connString)
    except Exception as e:
        print(e)
        # print('task is terminated')
        sys.exit()
    else:
        cursor = conn.cursor()

    n = len(table_fields)

    def deco(n):
        if n[0] == '[' and n[-1] == ']':
            return n
        else:
            return '[' + n + ']'

    insert_statement = f"""
        INSERT INTO {table} ({', '.join(map(deco, table_fields))})
        OUTPUT Inserted.{pk_name}
        VALUES ({"".join('?, ' for i in range(n))[:-2]})        
    """

    #print(insert_statement)
    #print(record)

    try:
        result = cursor.execute(insert_statement, record)
        pk = result.fetchone()[0]

    except Exception as e:
        cursor.rollback()
        print(e)
        # print('transaction rolled back')

    else:
        # print('records inserted successfully')
        cursor.commit()
        cursor.close()

    return pk


def update_db(table, pk_name, pk_value, table_fields, record):
    """
    For a selected table, insert new records in our database

    Args:
        table        : str   | the name of the table
        pk_name      : str   | the name of the primary key field
        pk_value     : int   | the value of the primary key field to update
        table_fields : str[] | array of fields that will be updated by the insert command
        record       : array | (columns/fields) (or tuple) with content to be inserted
    """
    try:
        conn = odbc.connect(connString)
    except Exception as e:
        print(e)
        # print('task is terminated')
        sys.exit()
    else:
        cursor = conn.cursor()

    n = len(table_fields)

    def deco(n):
        if n[0] == '[' and n[-1] == ']':
            return n
        else:
            return '[' + n + '] = ?'

    update_statement = f"""
        UPDATE {table} 
        SET {', '.join(map(deco, table_fields))}
        WHERE {pk_name} = {pk_value}
    """

    # print(update_statement)
    # print(record)

    try:
        result = cursor.execute(update_statement, record)

    except Exception as e:
        cursor.rollback()
        print(e.value)
        # print('transaction rolled back')

    else:
        # print('records updated successfully')
        cursor.commit()
        cursor.close()