from purikura_test import camera


def test_discover_cameras_does_not_reopen_active_camera(monkeypatch) -> None:
    opened_ids: list[int] = []

    class FakeCapture:
        def __init__(self, camera_id: int) -> None:
            self.camera_id = camera_id
            opened_ids.append(camera_id)

        def isOpened(self) -> bool:
            return self.camera_id == 1

        def read(self):
            return True, object()

        def release(self) -> None:
            return None

    monkeypatch.setattr(camera.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(camera.time, "sleep", lambda _: None)

    cameras = camera.discover_cameras(max_index=3, active_camera_id=0)

    assert opened_ids == [1, 2]
    assert cameras[0].id == 0
    assert cameras[0].available is True
    assert cameras[1].available is True
    assert cameras[2].available is False
