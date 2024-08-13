import os
import sqlite3
import dbMsSql as sql
import networkx as nx
import networkx.algorithms.approximation.maxcut as classical_maxcut
import shutil
import ast # used to transform string to 'list of tuples'

# move Dante sqlite3 file to local Sql Server file QuantumMaxCut database

sourceDanteSqlFolder = 'C:\Bikini Atoll\QUANTUM\Dante Results\_out20240713'
archiveDanteSqlFolder = sourceDanteSqlFolder + '/_archive'

# loop Dante sqlite3 folder for each db
# iterating over all files
for dbname in os.listdir(sourceDanteSqlFolder):
    if dbname.endswith('.db'):
        print(dbname)  # printing file name
        #print(dbname[7:22])

        # open Dante sqlite3
        conn = sqlite3.connect(f'{sourceDanteSqlFolder}/{dbname}')

        # tb_Graph
        cG = conn.cursor()
        cG.row_factory = sqlite3.Row
        cG.execute('SELECT * FROM tb_Graph') # LIMIT 10
        for rG in cG:
            gdante_fk = rG['graph_pk']
            graph_pk = sql.insert_db('tb_Graph', 'graph_pk'
                                    , ('DanteDb'   , 'Dante_fk' , '_date'    , 'graph_nodes'    , 'graph_edges'    ,'graph_is_weighted'     , 'graph_weight'    , 'graph_edge_list'    , 'graph_degree'    ,'graph_average_degree'     , 'graph_average_degree_connectivity'    , 'graph_density'    ,'graph_clustering'     , 'graph_average_clustering'    , 'graph_average_geodesic_distance'    ,'graph_betweenness_centrality'    , 'graph_average_betweenness_centrality'    )
                                    , (dbname, gdante_fk  , rG['_date'], rG['graph_nodes'], rG['graph_edges'], rG['graph_is_weighted'], rG['graph_weight'], rG['graph_edge_list'], rG['graph_degree'], rG['graph_average_degree'], rG['graph_average_degree_connectivity'], rG['graph_density'], rG['graph_clustering'], rG['graph_average_clustering'], rG['graph_average_geodesic_distance'],rG['graph_betweenness_centrality'], rG['graph_average_betweenness_centrality'])
                                    )
            # tb_Test
            cT = conn.cursor()
            cT.row_factory = sqlite3.Row
            cT.execute(f'SELECT * FROM tb_Test WHERE graph_fk={gdante_fk}') # LIMIT 10
            for rT in cT:
                tdante_fk = rT['test_pk']
                test_pk = sql.insert_db('tb_Test', 'test_pk'
                                         , ('DanteDb', 'Dante_fk' , '_date'     , 'execution_ref'    , 'n'    , 'graph_fk', 'maximize'    , 'poly_problem'    , 'angle_study'    ,  'layers'    , 'initial_angles_type'    , 'coefficients_type'    , 'pubo_variables'    , 'qubo_variables'    , 'qubits'    , 'ancillary_count'    , 'cost_function'    , 'execution_time'    , 'circuit_depth'   , 'circuit_depth_cx'    , 'circuit_depth_cx_parallel'    , 'transpiled_circuit_depth'    , 'transpiled_circuit_depth_cx'    , 'transpiled_circuit_depth_cx_parallel'     , 'result_bitstring'    , 'nfev'    , 'result'    , 'probability'    , 'expectation'    , 'final_mixer_angles'    , 'final_cost_angles'    , 'classical_call_count')
                                         , (dbname   , tdante_fk  , rT['_date'] , rT['execution_ref'], rT['n'], graph_pk  , rT['maximize'], rT['poly_problem'], rT['angle_study'],  rT['layers'], rT['initial_angles_type'], rT['coefficients_type'], rT['pubo_variables'], rT['qubo_variables'], rT['qubits'], rT['ancillary_count'], rT['cost_function'], rT['execution_time'], rT['circuit_depth'], rT['circuit_depth_cx'], rT['circuit_depth_cx_parallel'], rT['transpiled_circuit_depth'], rT['transpiled_circuit_depth_cx'], rT['transpiled_circuit_depth_cx_parallel'], rT['result_bitstring'], rT['nfev'], rT['result'], rT['probability'], rT['expectation'], rT['final_mixer_angles'], rT['final_cost_angles'], rT['classical_call_count'])
                                         )

                '''
                # tb_Test_Angle
                cAn = conn.cursor()
                cAn.row_factory = sqlite3.Row
                cAn.execute(f'SELECT * FROM tb_Test_Angle WHERE test_fk={tdante_fk}')
                for rAn in cAn:
                    sql.insert_db('tb_Test_Angle', 'testangle_pk'
                                  , ('Dante_fk'         , 'test_fk', 'minimize_iteration'     , 'expectation'     , 'angle_string'      )
                                  , (rAn['testangle_pk'], test_pk  , rAn['minimize_iteration'], rAn['expectation'], rAn['angle_string'])
                                  )
                cAn.close()
                '''

            cT.close()

        cG.close()
        conn.close()

        # move Dante sqlite3 to archive folder
        shutil.move(f'{sourceDanteSqlFolder}/{dbname}', f'{archiveDanteSqlFolder}/{dbname}')
