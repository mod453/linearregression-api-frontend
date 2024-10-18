from fastapi import FastAPI, File, UploadFile
import pandas as pd
import io
from fastapi.responses import StreamingResponse
import matplotlib.pyplot as plt
import numpy as np

app = FastAPI()

w = 1.984053
b = 3.00805

@app.post("/plot")
async def plot(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    # Ensure the predictions column exists in the DataFrame
    if "predictions" not in df.columns:
        df["predictions"] = b * df["inputs"] + b

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['inputs'], df['targets'], color="blue", label="Actual targets", marker='x')
    plt.plot(df['inputs'], df['predictions'], color="red", label="Predicted line")

    # Calculate Mean Squared Error (MSE) and display it on the plot
    mse_score = np.mean((df['predictions'].values - df['targets'].values) ** 2)
    plt.title(f"Linear Regression for Star Size Prediction (MSE: {mse_score:.2f})", color="red", fontsize=14)
    plt.xlabel("Brightness", color="red", fontsize=14)
    plt.ylabel("Star Size", color="red", fontsize=14)
    plt.legend()

    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    return StreamingResponse(buf, media_type="image/png", headers={"Content-Disposition": "attachment; filename=plot.png"})
