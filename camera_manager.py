import cv2
import platform


def _try_open_camera(source, api_preference):
    cap = cv2.VideoCapture(source, api_preference)
    if cap.isOpened():
        ret, _ = cap.read()
        if ret:
            return cap
    cap.release()
    return None


def open_camera_source(source):
    """Abrir fonte de câmera com um backend alternativo no Windows."""
    if isinstance(source, int) and platform.system() == "Windows":
        backends = [
            getattr(cv2, "CAP_MSMF", None),
            getattr(cv2, "CAP_DSHOW", None),
            getattr(cv2, "CAP_ANY", None),
        ]
        for api in [api for api in backends if api is not None]:
            try:
                cap = _try_open_camera(source, api)
            except Exception:
                cap = None

            if cap is not None:
                return cap

    return cv2.VideoCapture(source)


def list_cameras(max_test=10, exclude_indices=None):
    available = []
    exclude_indices = set(exclude_indices or [])

    for i in range(max_test):
        if i in exclude_indices:
            continue

        cap = open_camera_source(i)
        if cap is not None and cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
            cap.release()

    return available
