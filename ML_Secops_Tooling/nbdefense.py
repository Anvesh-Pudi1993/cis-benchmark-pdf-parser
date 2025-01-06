import os
import subprocess
import tempfile
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask
import uvicorn

app = FastAPI(
    docs_url="/nbdefense/api/docs",
    redoc_url="/nbdefense/api/redoc",
    openapi_url="/nbdefense/api/openapi.json"
)

def cleanup(temp_file: str):
    """Clean up temporary files."""
    try:
        os.remove(temp_file)
    except FileNotFoundError:
        pass

@app.post("/nbdefense/api/scan-notebook/")
async def scan_notebook(file: UploadFile = File(...)):
    """
    Scan uploaded `.ipynb` notebook files for security issues.
    """
    if not file.filename.endswith(".ipynb"):
        raise HTTPException(status_code=400, detail="Only .ipynb files are allowed")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file_path = os.path.join(temp_dir, file.filename)
            
            # Save the uploaded file to the temporary directory
            with open(input_file_path, "wb") as temp_file:
                temp_file.write(await file.read())

            # Generate a unique report filename
            report_file = os.path.join(temp_dir, "output_" + str(uuid.uuid4()) + ".html")

            # Run the nbdefense scanning tool
            try:
                result = subprocess.run(
                    ["nbdefense", "scan", input_file_path, "-f", report_file],
                    check=True,
                    capture_output=True,
                    text=True
                )
            except FileNotFoundError:
                raise HTTPException(status_code=500, detail="nbdefense is not installed or not in PATH")
            except subprocess.CalledProcessError as e:
                raise HTTPException(status_code=500, detail=f"nbdefense error: {e.stderr}")

            # Return the generated report
            return FileResponse(
                path=report_file,
                media_type="text/html",
                filename="nbdefense_output.html",
                background=BackgroundTask(cleanup, report_file)
            )
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(ex)}")

@app.get("/")
def get_response():
    """Basic health check endpoint."""
    return {"message": "Server is running"}

if __name__ == "__main__":
    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=8000)