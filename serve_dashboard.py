"""Serve the MedQuery dashboard alongside the backend for live verification.

FastAPI has no static file mount by default, so this mounts the dashboard
HTML on the same origin (port 8000) at /medquery-dashboard.html. With the
origin-relative API base URL, the dashboard now works on any host/port,
not just localhost.
"""
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from main import app

app.get("/medquery-dashboard.html")(lambda: FileResponse("medquery-dashboard.html", media_type="text/html"))

if __name__ == "__main__":
    import uvicorn
    from config import HOST, PORT
    uvicorn.run("serve_dashboard:app", host=HOST, port=PORT, reload=False)
