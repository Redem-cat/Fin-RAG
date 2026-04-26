"""
用户认证模块
提供用户注册、登录、密码验证功能
使用 SQLite 数据库存储用户信息
"""
import hashlib
import secrets
from datetime import datetime, timedelta
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


class UserProfile(Base):
    """用户画像表"""
    __tablename__ = "user_profiles"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True)
    
    # 基本信息
    display_name = Column(String(50), default="")  # 显示名称
    timezone = Column(String(50), default="Asia/Shanghai")  # 时区
    bio = Column(Text, default="")  # 个人简介
    
    # 投资偏好
    investment_experience = Column(String(20), default="新手")  # 投资经验: 新手/初级/中级/高级
    risk_preference = Column(String(20), default="稳健型")  # 风险偏好: 保守型/稳健型/平衡型/激进型
    interested_areas = Column(Text, default="")  # 感兴趣领域，逗号分隔
    
    # 首次使用引导
    has_completed_onboarding = Column(Integer, default=0)  # 是否完成首次引导
    onboarding_step = Column(Integer, default=0)  # 引导步骤
    
    # 设置
    notification_enabled = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    user = relationship("User", backref="profile")


class SessionToken(Base):
    """会话 Token 表，用于持久化登录状态"""
    __tablename__ = "session_tokens"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    token = Column(String(64), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.now)

    user = relationship("User", backref="session_tokens")


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
        
        # 自动创建用户画像
        create_user_profile(new_user.id)
        
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
        
        # 自动创建用户画像（如果不存在）
        create_user_profile(user.id)
        
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
    """保存消息到数据库和 Markdown 文件"""
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
        
        # 保存到 Markdown 文件（用于调试）
        _save_message_to_markdown(conversation_id, role, content, sources, used_context)
        
        return True
    except Exception as e:
        session.rollback()
        return False
    finally:
        session.close()


def _save_message_to_markdown(
    conversation_id: int,
    role: str,
    content: str,
    sources: Optional[List[Dict]] = None,
    used_context: bool = True
):
    """将消息保存到 Markdown 文件"""
    import json
    from pathlib import Path
    
    # 创建 conversations 目录
    conv_dir = Path(__file__).parent.parent / "memory" / "conversations"
    conv_dir.mkdir(parents=True, exist_ok=True)
    
    # 每个对话一个文件
    conv_file = conv_dir / f"conv_{conversation_id}.md"
    
    # 构建消息内容
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    role_display = "**YOU**" if role == "user" else "**AI**"
    
    markdown_content = f"\n\n---\n### {role_display} - {timestamp}\n\n{content}\n"
    
    # 如果有检索来源，添加来源信息
    if sources and used_context and role == "assistant":
        markdown_content += "\n**Sources:**\n"
        for i, source in enumerate(sources, 1):
            similarity = source.get('similarity', 0)
            source_name = source.get('source', 'unknown')
            markdown_content += f"- [{i}] {source_name} (similarity: {similarity:.3f})\n"
    
    # 追加到文件
    with open(conv_file, 'a', encoding='utf-8') as f:
        f.write(markdown_content)


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


# =========================
# 用户画像管理功能
# =========================

def create_user_profile(user_id: int) -> Optional[int]:
    """创建用户画像"""
    init_db()
    session = SessionLocal()
    try:
        # 检查是否已存在
        existing = session.query(UserProfile).filter(UserProfile.user_id == user_id).first()
        if existing:
            return existing.id
        
        profile = UserProfile(user_id=user_id)
        session.add(profile)
        session.commit()
        return profile.id
    except Exception as e:
        session.rollback()
        return None
    finally:
        session.close()


def get_user_profile(user_id: int) -> Optional[Dict[str, Any]]:
    """获取用户画像"""
    init_db()
    session = SessionLocal()
    try:
        profile = session.query(UserProfile).filter(UserProfile.user_id == user_id).first()
        if not profile:
            return None
        
        return {
            "id": profile.id,
            "user_id": profile.user_id,
            "display_name": profile.display_name,
            "timezone": profile.timezone,
            "bio": profile.bio,
            "investment_experience": profile.investment_experience,
            "risk_preference": profile.risk_preference,
            "interested_areas": profile.interested_areas,
            "has_completed_onboarding": bool(profile.has_completed_onboarding),
            "onboarding_step": profile.onboarding_step,
            "notification_enabled": bool(profile.notification_enabled),
            "created_at": profile.created_at.isoformat() if profile.created_at else None,
            "updated_at": profile.updated_at.isoformat() if profile.updated_at else None
        }
    finally:
        session.close()


def update_user_profile(user_id: int, **kwargs) -> bool:
    """更新用户画像"""
    init_db()
    session = SessionLocal()
    try:
        profile = session.query(UserProfile).filter(UserProfile.user_id == user_id).first()
        if not profile:
            # 如果不存在，先创建
            profile = UserProfile(user_id=user_id)
            session.add(profile)
        
        # 更新字段
        allowed_fields = [
            "display_name", "timezone", "bio",
            "investment_experience", "risk_preference", "interested_areas",
            "has_completed_onboarding", "onboarding_step", "notification_enabled"
        ]
        
        for key, value in kwargs.items():
            if key in allowed_fields and hasattr(profile, key):
                setattr(profile, key, value)
        
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        return False
    finally:
        session.close()


def complete_onboarding(user_id: int) -> bool:
    """完成首次引导"""
    return update_user_profile(
        user_id,
        has_completed_onboarding=1,
        onboarding_step=3
    )


def needs_onboarding(user_id: int) -> bool:
    """检查是否需要完成首次引导"""
    profile = get_user_profile(user_id)
    if not profile:
        return True
    return not profile["has_completed_onboarding"]


# =========================
# 会话 Token 管理（持久化登录）
# =========================

TOKEN_EXPIRE_DAYS = 7


def create_session_token(user_id: int) -> str:
    """
    创建会话 Token，有效期 7 天

    Returns:
        token 字符串
    """
    init_db()
    session = SessionLocal()
    try:
        # 删除该用户的旧 token
        session.query(SessionToken).filter(SessionToken.user_id == user_id).delete()

        # 创建新 token
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(days=TOKEN_EXPIRE_DAYS)

        session_token = SessionToken(
            user_id=user_id,
            token=token,
            expires_at=expires_at
        )
        session.add(session_token)
        session.commit()

        return token
    finally:
        session.close()


def validate_session_token(token: str) -> Optional[Dict[str, Any]]:
    """
    验证会话 Token

    Returns:
        用户信息 dict，如果无效则返回 None
    """
    if not token:
        return None

    init_db()
    session = SessionLocal()
    try:
        # 清理过期 token
        session.query(SessionToken).filter(SessionToken.expires_at < datetime.now()).delete()
        session.commit()

        # 查找 token
        st = session.query(SessionToken).filter(SessionToken.token == token).first()
        if not st or st.expires_at < datetime.now():
            return None

        # 获取用户信息
        user = session.query(User).filter(User.id == st.user_id).first()
        if not user:
            return None

        return {
            "id": user.id,
            "username": user.username,
            "created_at": user.created_at.isoformat() if user.created_at else None
        }
    finally:
        session.close()


def delete_session_token(token: str) -> bool:
    """删除会话 Token（登出）"""
    if not token:
        return False

    init_db()
    session = SessionLocal()
    try:
        result = session.query(SessionToken).filter(SessionToken.token == token).delete()
        session.commit()
        return result > 0
    finally:
        session.close()


if __name__ == "__main__":
    # 测试
    init_db()
    print("数据库初始化完成")
