import sys
import sqlite3


class LocalDB:
    def __init__(self, filename: str):
        self._dbpath = f'{filename}.db'


    # 🅲🆁🅴🅰🆃🅴 🆂🆀🅻🅸🆃🅴3 🅳🅰🆃🅰🅱🅰🆂🅴
    def create_db(self):
        print(self._dbpath)
        conn = sqlite3.connect(self._dbpath)
        c = conn.cursor()

        sql = '''
              CREATE TABLE IF NOT EXISTS tb_Test
              ( [_date] [timestamp] DEFAULT CURRENT_TIMESTAMP,
                [test_pk] [integer] primary key AUTOINCREMENT,
                [execution_ref] [nvarchar](50) NULL,
                [n] [tinyint] NULL,
                [graph_fk] [integer] NULL,
                [maximize] [bit] NULL,
                [classical_optimiser] [nvarchar](25) NULL,
                [poly_problem] [tinyint] NULL,
                [multi_angle] [bit] NULL,
                [angle_study] [tinyint] NULL,
                [angle_count] [int] NULL,
                [layers] [tinyint] NULL,
                [initial_angles_type] [tinyint] NULL,
                [coefficients_type] [smallint] NULL,
                [pubo_variables] [int] NULL,
                [qubo_variables] [int] NULL,
                [qubits] [int] NULL,
                [ancillary_count] [int] NULL,
                [cost_function] [nvarchar](500) NULL,                    
                [execution_time] [real] NULL,
                [shuffle] [int] NULL,                    
                [circuit_depth] [int] NULL,
                [circuit_depth_cx] [int] NULL,
                [circuit_depth_cx_parallel] [int] NULL,
                [transpiled_circuit_depth] [int] NULL,
                [transpiled_circuit_depth_cx] [int] NULL,
                [transpiled_circuit_depth_cx_parallel] [int] NULL,
                [result_bitstring] [nvarchar](500) NULL,                   
                [nfev] [int] NULL,
                [result] [int] NULL,
                [result_optimal] [int] NULL,
                [probability] [real] NULL,
                [expectation] [real] NULL,
                [approximation_ratio] [real] NULL,
                [final_mixer_angles] [nvarchar](500) NULL,
                [final_cost_angles] [nvarchar](1000) NULL,
                [classical_call_count] int NULL
              )
              '''
        c.execute(sql)  # create tables

        sql = '''
              CREATE TABLE IF NOT EXISTS tb_Graph
              ( [_date] [timestamp] DEFAULT CURRENT_TIMESTAMP,
                [graph_pk] [integer] primary key AUTOINCREMENT,
                [graph_type] [integer] NULL,
                [graph_nodes] [int] NULL,
                [graph_edges] [int] NULL,                
                [graph_is_weighted] [bit] NULL,
                [graph_weight] [real] NULL,
                [graph_is_hyper] [bit] NULL,
                [graph_edge_list] [nvarchar](500) NULL,
                [graph_x] [nvarchar](500) NULL,
                [graph_z] [nvarchar](500) NULL,
                [graph_degree] [nvarchar](500) NULL,
                [graph_average_degree] [real] NULL,
                [graph_average_degree_connectivity] [nvarchar](500) NULL,
                [graph_density] [real] NULL,
                [graph_clustering] [nvarchar](500) NULL,
                [graph_average_clustering] [real] NULL,
                [graph_average_geodesic_distance] [real] NULL,
                [graph_betweenness_centrality] [nvarchar](500) NULL,
                [graph_average_betweenness_centrality] [real] NULL
              )
              '''
        c.execute(sql)  # create tables

        sql = '''
              CREATE TABLE IF NOT EXISTS tb_Test_Angle
              ( [testangle_pk] [integer] primary key AUTOINCREMENT,
                [test_fk] [integer] NULL,
                [minimize_iteration] [integer] NULL,
                [expectation] [real] NULL,
                [angle_string] [text] NULL
              )
              '''
        c.execute(sql)  # create tables


        conn.commit()
        conn.close()

    # 🅸🅽🆂🅴🆁🆃 🅸🅽🆃🅾 🆂🆀🅻🅸🆃🅴3 🅳🅰🆃🅰🅱🅰🆂🅴
    def insert_db(self, table: str, pk_name: str, table_fields: tuple, record: tuple):
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
            conn = sqlite3.connect(self._dbpath)
        except Exception as e:
            print(e)
            sys.exit()
        else:
            cursor = conn.cursor()

        n = len(table_fields)

        def deco(n):
            # decorate fields with []
            if n[0] == '[' and n[-1] == ']':
                return n
            else:
                return '[' + n + ']'

        insert_statement = f"""
            INSERT INTO {table} ({', '.join(map(deco, table_fields))})
            VALUES ({"".join('?, ' for i in range(n))[:-2]})        
        """

        #print(self._dbpath)
        #print(insert_statement)
        #print(record)

        pk = -1
        #try:
        if 1 == 1:
            result = cursor.execute(insert_statement, record)
            pk = cursor.lastrowid
            # print(pk)

        #except Exception as e:
        #    print(e)
            #cursor.rollback()
            #print('transaction rolled back')

        #else:
            # print('records inserted successfully')
            conn.commit()
            conn.close()

        return pk

    # 🆄🅿🅳🅰🆃🅴 🆂🆀🅻🅸🆃🅴3 🅳🅰🆃🅰🅱🅰🆂🅴
    def update_db(self, table, pk_name, pk_value, table_fields, record):
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
            conn = sqlite3.connect(self._dbpath)
        except Exception as e:
            print(e)
            sys.exit()
        else:
            cursor = conn.cursor()

        n = len(table_fields)

        def deco(n):
            if n[0] == '['and n[-1] == ']':
                return n
            else:
                return '[' + n + '] = ?'

        update_statement = f"""
            UPDATE {table} 
            SET {', '.join(map(deco, table_fields))}
            WHERE {pk_name} = {pk_value}
        """

        #print(update_statement)
        #print(record)
        #print(pk_value)

        try:
            result = cursor.execute(update_statement, record)

        except Exception as e:
            print(e)
            # cursor.rollback()
            # print('transaction rolled back')

        else:
            # print('records updated successfully')
            conn.commit()
            conn.close()

