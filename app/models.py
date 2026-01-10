from sqlalchemy import Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.sql import func
from .database import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    file_path = Column(String)
    user_id = Column(String, index=True)
    status = Column(String, default="PROCESSING") # PROCESSING, COMPLETED, FAILED
    upload_time = Column(DateTime(timezone=True), server_default=func.now())
    
    # Metadata extracted
    content_text = Column(Text, nullable=True) # Full extracted text
    summary = Column(Text, nullable=True)
    tags = Column(JSON, nullable=True)
    category = Column(String, nullable=True)
    embedding = Column(JSON, nullable=True) # Vector embedding
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename='{self.filename}', status='{self.status}')>"
