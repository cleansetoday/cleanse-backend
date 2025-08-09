
import uvicorn
import os

if __name__ == "__main__":
    # Use PORT environment variable or fallback to 8080 for local development
    port = int(os.environ.get("PORT", "8080"))
    print(f"🚀 Starting FastAPI server on 0.0.0.0:{port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
