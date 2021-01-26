from typing import List, Optional
from fastapi import FastAPI, Query
import uvicorn
import json
import random as rnd
from random import shuffle

import pandas as pd
import datetime as dt
import psycopg2 # pip install psycopg2-binary

import networkx as nx
import math
import geopandas as gpd
from shapely.geometry import Point

class db_link():
    def __init__(self):
        self.conn = ''
        self.cursor = ''

    def open_conn(self):
    
        try:  
            conn = psycopg2.connect(
            host="80.87.194.16", # 'localhost'
            database="enjoy_art_and_save_time_db",
            port = "5432",
            user="backend",
            password="Coolspot@123654")
            cursor = conn.cursor()
            
            # Print PostgreSQL version
            cursor.execute("SELECT version();")
            record = cursor.fetchone()
            res = "You are connected to - " + str(record)
            
            self.conn = conn
            self.cursor = cursor
            
        except (Exception, psycopg2.Error) as error :
            res = 'Error while connecting to PostgreSQL' + str(error)
        
        return res
    
    def close_conn(self):
        if(self.conn):
            self.cursor.close()
            self.conn.close()
            return "PostgreSQL connection is closed"
    
    def get_table(self, t):
    #   print ( conn.get_dsn_parameters(),"\n")
        query = "SELECT * from e_and_s_schema." + t
        df_results = pd.read_sql_query(query, self.conn)
        return df_results
    
    def get_table_cond(self, t, col, value):
    #   print ( conn.get_dsn_parameters(),"\n")
        if type(value)=='str':
             query = "SELECT * from e_and_s_schema.%s where %s=''%s''" %(t, col, value)
        else:
            query = "SELECT * from e_and_s_schema.%s where %s=%s" %(t, col, value)
        df_results = pd.read_sql_query(query, self.conn)
        return df_results


class sq_density():
    
    def __init__(self, conn_obj, filename = './data/dens_per_hour.csv',):
        df = pd.read_csv(filename, usecols=['id', 'date', 'cnt', 'hour', 'day'])
        df.date = pd.to_datetime(df.date)
        self.df = df
        
        _ = conn_obj.open_conn()
        self.df_sq_id = conn_obj.get_table('showplace_fishnet')
        _ = conn_obj.close_conn()
        
        self.ids_list = list()
        self.head_pop = list() # топ популярных мест
        self.tail_pop = list() # топ самых непопулярных мест
    
    def create_week_df(self):
        # преобразование данных, надо вынести в бд или в файл
        week_df = self.df[['id', 'date', 'cnt', 'hour']]
        week_df['weekday'] = week_df.date.apply(lambda x: x.weekday())
        week_df = week_df.groupby(['id', 'weekday', 'hour'])['cnt'].median().reset_index()
        week_df.cnt = week_df.cnt.apply(int)
        
        match_df = dens_obj.df_sq_id
        match_df.rename(columns={'fish_id': 'id'}, inplace = True)
        
        week_id_df = pd.merge(week_df, match_df, how='inner', on='id')
        
        df_pop = week_id_df.groupby('showplace_id')['cnt'].median()
        df_pop.sort_values(inplace=True, ascending=False)
        df_pop = pd.DataFrame(df_pop).reset_index()
        df_pop['rank_cnt'] = df_pop.cnt.rank(ascending=False)
        
        self.week_id_df = week_id_df.copy()
        self.df_pop = df_pop.copy()
    
    def update_actual_ids(self, conn_obj):
        # обновляет список актуальных id, которые можно отображать в приложении
        q = 'select showplace_id from e_and_s_schema.showplace_tbl where LENGTH(description_ru) > 1'
        
        _ = conn_obj.open_conn()
        self.ids_list = pd.read_sql_query(q, conn_obj.conn)['showplace_id'].values.tolist()
        _ = conn_obj.close_conn()
  
    def update_head_pop(self, part = 0.2):
        # part - доля самых популярных
        df_pop_act = self.df_pop[self.df_pop.showplace_id.isin(self.ids_list)]
        border_val = df_pop_act.rank_cnt.quantile(1 - part)
        self.head_pop = df_pop_act[df_pop_act.rank_cnt >= border_val].showplace_id.values.tolist()
        
    def update_tail_pop(self, part = 0.2):
        # part - доля самых непопулярных
        df_pop_act = self.df_pop[self.df_pop.showplace_id.isin(self.ids_list)]
        border_val = df_pop_act.rank_cnt.quantile(part)
        self.tail_pop = df_pop_act[df_pop_act.rank_cnt <= border_val].showplace_id.values.tolist()
    
    def set_db_popularuty(self, conn_obj):
        conn_obj.open_conn()
        for num, row in self.df_pop.iterrows():
            q = 'UPDATE e_and_s_schema.showplace_tbl SET popularity = ' + str(row['rank_cnt']) + ' WHERE showplace_id = ' + str(row['showplace_id'])
            conn_obj.cursor.execute(q)
            
        conn_obj.conn.commit()
        conn_obj.close_conn()
        
    def get_hour_cnt(self, target_date, hour, showplace_id):
        target_date = pd.to_datetime(target_date)
        weekday = target_date.weekday()
        res = self.week_id_df[(self.week_id_df.showplace_id == showplace_id) & 
                               (self.week_id_df.weekday == weekday) & 
                               (self.week_id_df.hour == hour)]['cnt']
        res = res.values[0] if len(res) == 1 else -1
        return res
    
    def get_list_between_dates(self, left, right, showplace_id):
        left = pd.to_datetime(left)
        right = pd.to_datetime(right)

        res_list = list()
        while left < right:
            left += np.timedelta64(1,'h')
            str_df = self.week_id_df[(self.week_id_df.showplace_id == showplace_id) & 
                                     (self.week_id_df.weekday == left.weekday()) & 
                                     (self.week_id_df.hour == left.hour)]
            cnt = -1
            if len(str_df) == 1:
                cnt = str_df.cnt.values[0]
            res_list.append((cnt, left))

        res_list.sort()
        res_list = [x[1] for x in res_list]
        return res_list
    
    def get_user_reccomend(self, conn_obj, user_id):
        user_id = str(user_id)
        q = '''
        select distinct ucoit.category_of_interest_id
        from e_and_s_schema.user_categories_of_interests_tbl ucoit
        where ucoit.user_id = ''' + user_id

        conn_obj.open_conn()
        conn_obj.cursor.execute(q)
        records = conn_obj.cursor.fetchall()

        if len(records) == 0:
            q = 'select distinct ucoit.category_of_interest_id from e_and_s_schema.user_categories_of_interests_tbl ucoit'
            conn_obj.cursor.execute(q)
            records = conn_obj.cursor.fetchall()
        records = [x[0] for x in records]
        records = '(' + str(records)[1:-1] + ')'
        q = '''
        select showplace_id from e_and_s_schema.showplace_tbl st 
        where 1=1
        and (
        st.k1 in <interests> or
        st.k2 in <interests> or
        st.k3 in <interests> or
        st.k4 in <interests> or
        st.k5 in <interests>
        )
        '''.replace('<interests>', records)
        conn_obj.cursor.execute(q)
        ids = conn_obj.cursor.fetchall()
        ids = [x[0] for x in ids]

        conn_obj.close_conn()
        
        return list(set(ids) & set(self.ids_list))
    
class user_anketa():
    
    def __init__(self, route_id, conn_obj):
        now = dt.datetime.today() # CONST
        
        self.route_id = route_id
        
        self.anketa=self.get_from_route_tbl(conn_obj)
        
        self.user_id  = self.anketa.iloc[0]['user_id']
        
        self.visit_time =  int ( self.anketa.iloc[0]['prof_type'] )
        self.wake_up_time = int (self.anketa.iloc[0]['wake_up_time']) +6 
        self.n_days = int( self.anketa.iloc[0]['count_days'] )
        
        
        if self.anketa.iloc[0]['date_start']  is None:
            self.first_day = now
        else:
            self.first_day = self.anketa.iloc[0]['date_start']
        
#         self.last_day = self.first_day + dt.timedelta (days=self.n_days)
        
        self.list_interests = self.get_interests(conn_obj)
        self.start_point = self.get_location(conn_obj)
        self.list_selected_places = self.get_selected_places(conn_obj)  

    def get_from_route_tbl (self, conn_obj):
#         try:
        _ = conn_obj.open_conn()
        df = conn_obj.get_table_cond('route_tbl', 'route_id',self.route_id)
        _ = conn_obj.close_conn()
         
        return df
       
#         except:
            
#             return 0
    def get_location(self, conn_obj):
        red_square_coords = (55.754356, 37.620173) #CONST
        
        _ = conn_obj.open_conn()
        df = conn_obj.get_table_cond('user_tbl', 'user_id', self.user_id)
        _ = conn_obj.close_conn()
        lat = df.iloc[0]['gps_lat']
        long = df.iloc[0]['gps_long']
        
        if lat==0 or long==0 or lat is None or long is None :
            return red_square_coords
        else:
            return lat, long

    def get_interests(self, conn_obj):
        
        _ = conn_obj.open_conn()
        df = conn_obj.get_table_cond('user_categories_of_interests_tbl', 'user_id', self.user_id)
        _ = conn_obj.close_conn()
        list_v=  df['category_of_interest_id'].values
        
        return list_v
    
    def get_selected_places(self, conn_obj):
    
        _ = conn_obj.open_conn()
        df = conn_obj.get_table_cond('route_object_tbl', 'route_id', self.route_id)
        _ = conn_obj.close_conn()
        list_v = df.loc[df['recommend_type']=='user choice']['showplace_id'].values
        return list_v

class itinerary():
    
    def __init__(self, ua):
        
        self.itinerary = list()
        self.obj_num = ua.n_days*4
        self.final_list = []
        self.object_dict=dict()
        self.interval=30
        
    def get_start(self, ua):
        
        if ua.first_day.hour<16:
            
            self.start_day=ua.first_day.date()
            self.start_hour=ua.first_day.hour+1
        else:
            
            self.start_day=ua.first_day.date()+dt.timedelta(days=1)
            self.start_hour=ua.wake_up_time
        
#         return start_day, start_hour
    def get_adj_mat(self,):
        adj_mat=nx.to_pandas_adjacency(self.G)
        dic_col={i:adj_mat.loc[adj_mat[i]!=0][i].mean()*math.pow(10,self.object_dict[i]) for i in adj_mat.columns}
        sorted_cols=list({k: v for k, v in sorted(dic_col.items(), key=lambda item: 100000 if item[1]/item[1]!=1  else item[1])}.keys())
        
        return adj_mat
    
    def get_dict_days(self, ua, dens_obj):
        
        adj_mat=self.get_adj_mat()
        left_list = adj_mat.columns
        
        for n_day in range( ua.n_days):
#             print(n_day)
            day = str(self.start_day + dt.timedelta(days=n_day))
            # self.itinerary [day] = list()
            
            comment=''
        #     print(left_list)
            if n_day==0:
                start_hour=self.start_hour
            else:
                if  ua.wake_up_time<9:
                    start_hour=9
                    comment='Вы встаете рано. Сходите вкусно позавтракайте'
                else:
                    start_hour=ua.wake_up_time
                    

            start_cur_hour=start_hour
            start_cur_min=0
            order=1
#             print('start_vis_hour',start_vis_hour)
            adj_mat= adj_mat.loc[left_list][left_list]
            while (start_cur_hour< 16 or (start_cur_hour==16 and start_cur_min<30)) and len(adj_mat.columns)>0 and order<4:
#                 print(order)
                if order!=1:
#                     print(adj_mat)
                    show_id, time_to_get = self.get_next_place(show_id, adj_mat, day, start_vis_hour, dens_obj)
                    adj_mat= adj_mat.loc[left_list][left_list]

                else:

                    time_to_get=60 # должна быть функиця расстояния от тек местоположения
                    adj_mat= adj_mat.loc[left_list][left_list]
                    show_id=adj_mat.columns[0]  

#                 print('куда как долго',show_id, time_to_get)
                start_vis_hour, start_vis_min, start_time_to_show= self.get_hour_min_start(start_cur_hour, start_cur_min,time_to_get)
               
                
                vis_time = int( min( ua.visit_time*60*2/3, ( (18 - start_vis_hour)*60 - start_vis_min )*2/3 ) )


                if start_vis_hour>18:
                    is_active=0 
                else:
                    is_active=1
                
                self.itinerary.append(
                    {
                    'id':str(show_id),
                    'order':order,
                    'time to get': int(time_to_get) ,
                    'time_to_start':  start_time_to_show,
                    'visit_time':vis_time ,
                    'comments': comment,
                    'is_active': is_active,
                    'day':n_day+1
                    }
                )
                # print(self.itinerary [day])
                order+=1
    
                start_cur_hour, start_cur_min, _= self.get_hour_min_start(start_vis_hour, start_vis_min, vis_time)

                left_list=[ i for i in left_list if i != show_id]
                comment=''
        #         print(left_list)
        #     self.obj_num+=order
            
    def get_random_dist(self):
        
        self.list_dist = list_distance
        h_v = len(self.list_dist)-1
        return self.list_dist[rnd.randint(0,h_v)]   
    
    def get_dist_between_start(start_location):
        return 0
            
    def get_hour_min_start(self, start_hour, start_min, time_to_get):
        
        start_vis_min = int((start_min + time_to_get%60 + self.interval)%60)
        ost = int((start_min+time_to_get%60 + self.interval)//60)
        start_vis_hour = int(start_hour + time_to_get//60 + ost)
    
        if len(str(start_vis_min))<2:
            
            str_min='0'+str(start_vis_min)
        else:
            str_min=str(start_vis_min)

        start_time_to_show=str(start_vis_hour)+':'+str_min
        
        return start_vis_hour,start_vis_min, start_time_to_show
    
    def get_list_ob(self, ua):
           
        self.object_dict = dict(zip(ua.list_selected_places,[1]*len(ua.list_selected_places)))
        
#         print(self.object_dict,len(self.object_dict))
        
        if len(self.object_dict)>=self.obj_num:
#             final_list = selected_list[:self.obj_num]
            return self.object_dict 

        else:
            inter=list(self.select_interesting(ua.list_interests))
#             print(inter)
            dic=dict(zip(inter,[2]*len(inter)))
            self.object_dict={**self.object_dict, **dic}
            
#         print(self.object_dict,len(self.object_dict))   
        
        if len(self.object_dict)>=self.obj_num:
            return self.object_dict
        
        else:
            n=self.obj_num-len(self.object_dict)
            pop=self.select_top_popular(n, list(self.object_dict.keys()))
            dic_pop=dict(zip(pop,[3]*len(pop)))
            self.object_dict={**self.object_dict, **dic_pop}
            
        return self.object_dict
        
    def select_top_popular(self, n, final_list):
        
        _ = conn_obj.open_conn()
        df = conn_obj.get_table('showplace_tbl')
        _ = conn_obj.close_conn()
        df=df.sort_values('popularity', ascending=False)
        df=df[~df['showplace_id'].isin(final_list)]
        return df.iloc[:n]['showplace_id'].values.tolist()
    
    def get_neigbors_by_iso(self, show_id, iso_type):
         
        _ = conn_obj.open_conn()
        df = conn_obj.get_table_cond('neighboring_showplace_tbl', 'showplace_id', show_id)
        _ = conn_obj.close_conn()
        val=df[df['isochrone_type']==iso_type].values.tolist()
        return val
    
    def get_iso_by_showplace(self, show_id1, show_id2):
        
        _ = conn_obj.open_conn()
        df = conn_obj.get_table_cond('neighboring_showplace_tbl', 'showplace_id', show_id1)
        _ = conn_obj.close_conn()
        iso_list=[iso for lis, iso in zip(df['list_showplaces'],df['isochrone_type']) if show_id2 in lis]
        if len(iso_list)>0:
            return iso_list[0]
        else:
            return  None
    
    def select_interesting (self,  list_interests):
        
        _ = conn_obj.open_conn()
        df = conn_obj.get_table('showplace_tbl')
        _ = conn_obj.close_conn()
        
        ind = (
                [ i for i,v in df['k1'].items() if v in list_interests ]
                +
                [ i for i,v in df['k2'].items() if v in list_interests ]
                +
                [ i for i,v in df['k3'].items() if v in list_interests ]
                +
                [ i for i,v in df['k4'].items() if v in list_interests ]
                +
                [ i for i,v in df['k5'].items() if v in
                 list_interests ]
               )
#         ind=[i for i in ind_1 if i in ind_2]
        
        list_obj = df.loc[ind]['showplace_id'].values.tolist()
        
        return list_obj
    
    def get_coords ( self, list_obj):
        _ = conn_obj.open_conn()
        df = conn_obj.get_table('showplace_tbl')
        _ = conn_obj.close_conn()
        list_obj = df[df['showplace_id'].isin(list_obj)]
    
    def get_graph(self):
        
        self.final_list=list(self.object_dict.keys())
        
        list_gr = [(self.final_list[i],self.final_list[j],self.get_iso_by_showplace(self.final_list[i],self.final_list[j])) for i in range(len(self.final_list)) for j in range(i+1,len(self.final_list))]
        list_gr1 = [ i for i in list_gr]
        self.G = nx.Graph()
        self.G.add_weighted_edges_from(list_gr1)
#         print(self.G.edges())
#         return G
    
    def get_next_place(self, show_id, adj_mat, date, hour, dens_obj):
        
        list_next_var=adj_mat[adj_mat[show_id]>0][show_id].index
#         print('list_next_var',list_next_var)
        if len(list_next_var)> 0:

            sel_list=[i for i in list_next_var if self.object_dict[i] == 1]
            if len(sel_list)>0:
                next_show=sel_list[0]   
            else:
                
                dict_dens={ dens_obj.get_hour_cnt(date, hour, i):i  for i in  list_next_var }
                
                next_show=dict_dens[min(dict_dens.keys())]

            time_to_get=adj_mat.loc[next_show][show_id]

        else:
            next_show=[i for i in adj_mat.columns if i!=show_id ][0]
            time_to_get=90

        return next_show, time_to_get
        
        
class CONST_VAL():
    # класс для хранения глобальных переменных
    def __init__(self):
        self.lunch_time = 2


app = FastAPI()

conn_obj = db_link()
dens_obj = sq_density(conn_obj)

dens_obj.create_week_df()
dens_obj.update_actual_ids(conn_obj)
dens_obj.update_head_pop()
dens_obj.update_tail_pop()

# def generate_rnd_ids():
#     return rnd.sample(dens_obj.ids_list, k = int(len(dens_obj.ids_list) * 0.2))


# @app.put("/update_actual_obj_id")
# async def read_items(q: Optional[List[int]] = Query([1, 10, 201, 239])):
#     const_v.upd_borders(q)
#     return q
@app.get('/place_for_you')
def get_place4you(id: int):
    return dens_obj.get_user_reccomend(conn_obj, id)
    
@app.get('/popular_places')
def get_popular_places():
    return dens_obj.head_pop
    
@app.get('/non_popular_places')
def get_non_popular_places():
    return dens_obj.tail_pop


@app.get('/route')
def get_route(route_id: int):
    ua = user_anketa(route_id, conn_obj)
    it = itinerary(ua)
    it.get_start(ua)
    it.get_list_ob(ua)
    it.get_graph()
    it.get_dict_days(ua, dens_obj)

    return it.itinerary

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

