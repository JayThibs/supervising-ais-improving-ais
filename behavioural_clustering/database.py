from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

engine = create_engine("sqlite:///statements.db")
Session = sessionmaker(bind=engine)

Base = declarative_base()


class Statement(Base):
    __tablename__ = "statements"

    id = Column(Integer, primary_key=True)
    text = Column(String)
    cluster = Column(Integer)
    labels = Column(String)  # JSON array
    toxicity = Column(Float)

    def __repr__(self):
        return f"Statement(text='{self.text}')"


Base.metadata.create_all(engine)


def load_statements_from_db(query):
    # Query DB for statements
    session = Session()
    statements = session.query(Statement).filter(Statement.text.contains(query))
    return statements
