function initMap(lat, lng, zoom, savePath) {
    // Use the provided lat, lng, and zoom or use default values
    const map = new google.maps.Map(document.getElementById("map"), {
        zoom: zoom || 13,
        center: { lat: lat || 34.04924594193164, lng: lng || -118.24104309082031 },
    });

    const trafficLayer = new google.maps.TrafficLayer();
    trafficLayer.setMap(map);

    // Save the result to the specified savePath
    const result = {
        lat: map.getCenter().lat(),
        lng: map.getCenter().lng(),
        zoom: map.getZoom(),
    };

    // Assuming savePath is a function that will handle saving the result
    // if (savePath && typeof savePath === 'function') {
    //     savePath(result);
    // }
}

// Example usage:
// Replace 'YOUR_SAVE_PATH_FUNCTION' with the actual function you want to use for saving
// e.g., initMap(34.04924594193164, -118.24104309082031, 15, YOUR_SAVE_PATH_FUNCTION);
// latitude = 13.838724
// longitude = 100.575318
