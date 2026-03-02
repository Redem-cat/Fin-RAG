"""
用户认证模块
提供用户注册、登录、密码验证功能
使用 SQLite 数据库存储用户信息
"""
import hashlib
import secrets
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# 数据库路径
DB_PATH = Path(__file__).parent.parent / "data" / "users.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# 数据库连接
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class User(Base):
    """用户表"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    password_hash = Column(String(64), nullable=False)
    salt = Column(String(16), nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    
    # 关联对话
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")


class Conversation(Base):
    """对话记录表"""
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(200), default="新对话")
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    """消息记录表"""
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    role = Column(String(20), nullable=False)  # user / assistant
    content = Column(Text, nullable=False)
    sources = Column(Text, nullable=True)  # JSON 存储检索来源
    used_context = Column(Integer, default=1)  # 是否使用上下文
    created_at = Column(DateTime, default=datetime.now)
    
    conversation = relationship("Conversation", back_populates="messages")


# 初始化数据库
def init_db():
    """初始化数据库表"""
    Base.metadata.create_all(bind=engine)


def hash_password(password: str, salt: str) -> str:
    """密码哈希"""
    return hashlib.sha256((password + salt).encode()).hexdigest()


def generate_salt() -> str:
    """生成随机盐值"""
    return secrets.token_hex(8)


def register(username: str, password: str) -> tuple[bool, str]:
    """
    用户注册
    返回: (成功标志, 消息)
    """
    init_db()
    session = SessionLocal()
    try:
        # 检查用户名是否已存在
        existing = session.query(User).filter(User.username == username).first()
        if existing:
            return False, "用户名已存在"
        
        # 创建新用户
        salt = generate_salt()
        password_hash = hash_password(password, salt)
        
        new_user = User(
            username=username,
            password_hash=password_hash,
            salt=salt
        )
        session.add(new_user)
        session.commit()
        
        return True, "注册成功"
    except Exception as e:
        session.rollback()
        return False, f"注册失败: {str(e)}"
    finally:
        session.close()


def login(username: str, password: str) -> tuple[bool, Optional[Dict[str, Any]], str]:
    """
    用户登录
    返回: (成功标志, 用户信息, 消息)
    """
    init_db()
    session = SessionLocal()
    try:
        user = session.query(User).filter(User.username == username).first()
        if not user:
            return False, None, "用户名或密码错误"
        
        password_hash = hash_password(password, user.salt)
        if password_hash != user.password_hash:
            return False, None, "用户名或密码错误"
        
        user_info = {
            "id": user.id,
            "username": user.username,
            "created_at": user.created_at.isoformat()
        }
        return True, user_info, "登录成功"
    finally:
        session.close()


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """根据ID获取用户信息"""
    init_db()
    session = SessionLocal()
    try:
        user = session.query(User).filter(User.id == user_id).first()
        if user:
            return {
                "id": user.id,
                "username": user.username,
                "created_at": user.created_at.isoformat()
            }
        return None
    finally:
        session.close()


# =========================
# 对话管理功能
# =========================

def create_conversation(user_id: int, title: str = "新对话") -> Optional[int]:
    """创建新对话，返回对话ID"""
    init_db()
    session = SessionLocal()
    try:
        conversation = Conversation(user_id=user_id, title=title)
        session.add(conversation)
        session.commit()
        return conversation.id
    except Exception as e:
        session.rollback()
        return None
    finally:
        session.close()


def save_message(
    conversation_id: int,
    role: str,
    content: str,
    sources: Optional[List[Dict]] = None,
    used_context: bool = True
) -> bool:
    """保存消息到数据库"""
    import json
    
    init_db()
    session = SessionLocal()
    try:
        sources_json = json.dumps(sources, ensure_ascii=False) if sources else None
        
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            sources=sources_json,
            used_context=1 if used_context else 0
        )
        session.add(message)
        
        # 更新对话更新时间
        conversation = session.query(Conversation).filter(Conversation.id == conversation_id).first()
        if conversation:
            conversation.updated_at = datetime.now()
        
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        return False
    finally:
        session.close()


def get_user_conversations(user_id: int) -> List[Dict[str, Any]]:
    """获取用户的所有对话列表"""
    init_db()
    session = SessionLocal()
    try:
        conversations = session.query(Conversation).filter(
            Conversation.user_id == user_id
        ).order_by(Conversation.updated_at.desc()).all()
        
        result = []
        for conv in conversations:
            # 获取最后一条消息作为预览
            last_message = session.query(Message).filter(
                Message.conversation_id == conv.id
            ).order_by(Message.created_at.desc()).first()
            
            result.append({
                "id": conv.id,
                "title": conv.title,
                "created_at": conv.created_at.isoformat(),
                "updated_at": conv.updated_at.isoformat(),
                "last_message_preview": last_message.content[:50] + "..." if last_message and len(last_message.content) > 50 else (last_message.content if last_message else "")
            })
        
        return result
    finally:
        session.close()


def get_conversation_messages(conversation_id: int) -> List[Dict[str, Any]]:
    """获取指定对话的所有消息"""
    init_db()
    session = SessionLocal()
    try:
        messages = session.query(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(Message.created_at.asc()).all()
        
        import json
        result = []
        for msg in messages:
            sources = json.loads(msg.sources) if msg.sources else None
            result.append({
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "sources": sources,
                "used_context": bool(msg.used_context),
                "created_at": msg.created_at.isoformat()
            })
        
        return result
    finally:
        session.close()


def delete_conversation(conversation_id: int, user_id: int) -> bool:
    """删除对话（需要验证用户身份）"""
    init_db()
    session = SessionLocal()
    try:
        conversation = session.query(Conversation).filter(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id
        ).first()
        
        if not conversation:
            return False
        
        session.delete(conversation)
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        return False
    finally:
        session.close()


def update_conversation_title(conversation_id: int, user_id: int, title: str) -> bool:
    """更新对话标题"""
    init_db()
    session = SessionLocal()
    try:
        conversation = session.query(Conversation).filter(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id
        ).first()
        
        if not conversation:
            return False
        
        conversation.title = title
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        return False
    finally:
        session.close()


if __name__ == "__main__":
    # 测试
    init_db()
    print("数据库初始化完成")
