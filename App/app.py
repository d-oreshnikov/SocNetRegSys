import os
import pandas as pd

from typing import List, Tuple
from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from schema import PostGet
from catboost import CatBoostClassifier
from datetime import datetime


SQLALCHEMY_DATABASE_URL = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"
USER_TABLE_QUERY = 'SELECT * FROM "d.oreshnikov_user_table_3"'
POST_TABLE_QUERY = 'SELECT * FROM "d.oreshnikov_post_table_3"'


engine = create_engine(SQLALCHEMY_DATABASE_URL,  pool_size=20, max_overflow=20)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()



class Post(Base):
    __tablename__ = "post"


    id = Column(Integer, primary_key=True, name="id")
    text = Column(String, name = "text")
    topic = Column(String, name= "topic")


def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_models():
    model_path = get_model_path("catboost_model")    
    from_file = CatBoostClassifier()
    return from_file.load_model(model_path, format='cbm')




def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)



def load_features()->  Tuple[pd.DataFrame, pd.DataFrame]:

    user_df = batch_load_sql(USER_TABLE_QUERY)
    post_df = batch_load_sql(POST_TABLE_QUERY)
    return user_df, post_df



def get_db():
    with SessionLocal() as db:
        return db



model = load_models()
user_df, post_df = load_features()

app = FastAPI()



@app.get("/post/recommendations/", response_model=List[PostGet])
def get_post_recommendations(id : int, time : datetime, limit : int = 10, db : Session =  Depends(get_db)):

    user = user_df[user_df["user_id"] == id]
    df_ready = post_df.merge(user, how="cross").drop("user_id", axis=1) 
    post_id = pd.concat([df_ready['post_id'], pd.DataFrame(model.predict_proba(df_ready).T[1], columns=['prediction'])], axis=1)\
                .sort_values(by=['prediction'], ascending=False).head(limit)['post_id'].values
    
    return db.query(Post).filter(Post.id.in_([int(i) for i in post_id])).limit(limit).all()