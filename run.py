from Foodimg2Ing import app

if __name__ == "__main__":
    import os

    # Render provides the port via the PORT env var. Default to 10000.
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
