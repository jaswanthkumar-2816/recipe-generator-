from Foodimg2Ing import app

if __name__ == "__main__":
    # Bind to 0.0.0.0 so the app is reachable via a tunnel (e.g., ngrok)
    # Use 5001 because macOS ControlCenter/AirPlay may occupy 5000.
    app.run(host="0.0.0.0", port=5001, debug=True)
