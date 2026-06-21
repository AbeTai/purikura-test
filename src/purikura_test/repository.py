from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from sqlalchemy import DateTime, ForeignKey, Integer, LargeBinary, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

from purikura_test.api_models import CaptureSummary, EffectSettings, FrameSummary


class Base(DeclarativeBase):
    pass


class CaptureRecord(Base):
    __tablename__ = "captures"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    camera_id: Mapped[int] = mapped_column(Integer, nullable=False)
    effect_settings_json: Mapped[str] = mapped_column(String, nullable=False)
    frame_id: Mapped[int | None] = mapped_column(ForeignKey("frames.id"), nullable=True)
    image_mime: Mapped[str] = mapped_column(String, nullable=False)
    image_blob: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)


class FrameRecord(Base):
    __tablename__ = "frames"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    image_mime: Mapped[str] = mapped_column(String, nullable=False)
    image_blob: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)


class CaptureRepository:
    def __init__(self, database_url: str = "sqlite:///data/purikura.sqlite3") -> None:
        if database_url.startswith("sqlite:///"):
            db_path = Path(database_url.removeprefix("sqlite:///"))
            if str(db_path) != ":memory:":
                db_path.parent.mkdir(parents=True, exist_ok=True)
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        engine_kwargs = {"connect_args": connect_args}
        if database_url == "sqlite:///:memory:":
            engine_kwargs["poolclass"] = StaticPool
        self._engine = create_engine(database_url, **engine_kwargs)
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)

    def init_schema(self) -> None:
        Base.metadata.create_all(self._engine)

    def add_frame(self, *, name: str, image_blob: bytes, image_mime: str = "image/png") -> FrameSummary:
        with self._session_factory() as session:
            record = FrameRecord(name=name, image_blob=image_blob, image_mime=image_mime)
            session.add(record)
            session.commit()
            return FrameSummary.model_validate(record)

    def list_frames(self) -> list[FrameSummary]:
        with self._session_factory() as session:
            records = session.scalars(select(FrameRecord).order_by(FrameRecord.created_at.desc())).all()
            return [FrameSummary.model_validate(record) for record in records]

    def get_frame_blob(self, frame_id: int) -> tuple[bytes, str] | None:
        with self._session_factory() as session:
            record = session.get(FrameRecord, frame_id)
            if record is None:
                return None
            return record.image_blob, record.image_mime

    def add_capture(
        self,
        *,
        camera_id: int,
        settings: EffectSettings,
        frame_id: int | None,
        image_blob: bytes,
        image_mime: str,
        width: int,
        height: int,
    ) -> CaptureSummary:
        with self._session_factory() as session:
            record = CaptureRecord(
                camera_id=camera_id,
                effect_settings_json=json.dumps(settings.model_dump(), separators=(",", ":")),
                frame_id=frame_id,
                image_mime=image_mime,
                image_blob=image_blob,
                width=width,
                height=height,
            )
            session.add(record)
            session.commit()
            return CaptureSummary.model_validate(record)

    def list_captures(self, limit: int = 50) -> list[CaptureSummary]:
        with self._session_factory() as session:
            statement = select(CaptureRecord).order_by(CaptureRecord.created_at.desc()).limit(limit)
            records = session.scalars(statement).all()
            return [CaptureSummary.model_validate(record) for record in records]

    def get_capture_image(self, capture_id: int) -> tuple[bytes, str] | None:
        with self._session_factory() as session:
            record = session.get(CaptureRecord, capture_id)
            if record is None:
                return None
            return record.image_blob, record.image_mime

    def session(self) -> Session:
        return self._session_factory()
