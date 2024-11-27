import requests
import math
from dotenv import load_dotenv
import os


def latLngToPoint(mapWidth, mapHeight, lat, lng):

    x = (lng + 180) * (mapWidth / 360)
    y = (
        (
            1
            - math.log(
                math.tan(lat * math.pi / 180) + 1 / math.cos(lat * math.pi / 180)
            )
            / math.pi
        )
        / 2
    ) * mapHeight

    return (x, y)


def pointToLatLng(mapWidth, mapHeight, x, y):

    lng = x / mapWidth * 360 - 180

    n = math.pi - 2 * math.pi * y / mapHeight
    lat = 180 / math.pi * math.atan(0.5 * (math.exp(n) - math.exp(-n)))

    return (lat, lng)


def getImageBounds(mapWidth, mapHeight, xScale, yScale, lat, lng):

    centreX, centreY = latLngToPoint(mapWidth, mapHeight, lat, lng)

    southWestX = centreX - (mapWidth / 2) / xScale
    southWestY = centreY + (mapHeight / 2) / yScale
    SWlat, SWlng = pointToLatLng(mapWidth, mapHeight, southWestX, southWestY)

    northEastX = centreX + (mapWidth / 2) / xScale
    northEastY = centreY - (mapHeight / 2) / yScale
    NElat, NElng = pointToLatLng(mapWidth, mapHeight, northEastX, northEastY)

    return [SWlat, SWlng, NElat, NElng]


def getLatStep(mapWidth, mapHeight, yScale, lat, lng):

    pointX, pointY = latLngToPoint(mapWidth, mapHeight, lat, lng)

    steppedPointY = pointY - ((mapHeight) / yScale)
    newLat, originalLng = pointToLatLng(mapWidth, mapHeight, pointX, steppedPointY)

    latStep = lat - newLat

    return latStep


def requestImage(
    picHeight, picWidth, zoom, scale, maptype, lat, lng, row, col, AreaID="AreaID"
):

    center = str(lat) + "," + str(lng)
    url = (
        "https://maps.googleapis.com/maps/api/staticmap?center="
        + center
        + "&zoom="
        + str(zoom)
        + "&size="
        + str(picWidth)
        + "x"
        + str(picHeight)
        + "&key="
        + os.getenv("api_key")
        + "&maptype="
        + maptype
        + "&scale="
        + str(scale)
    )
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = f"{dir_path}/output/" + AreaID + str(col) + "," + str(row) + ".png"
    folder = os.path.dirname(filename)
    os.makedirs(folder, exist_ok=True)
    r = requests.get(url)
    assert r.status_code == 200, f"status code: {r.status_code}"
    f = open(filename, "wb")
    f.write(r.content)
    f.close()

    print("writtern to file: " + filename)


def main():
    load_dotenv()
    # Bounding box for area to be scanned. AreaID is added to file name.
    AreaID = "SanFran"
    # center: { lat: 13.91365211, lng: 99.67114584 },
    padding = 0.05

    # northWestLat = 37.806716
    # northWestLng = -122.477702
    # southEastLat = 37.7636132
    # southEastLng = -122.4319237

    northWestLat = 13.91365211
    northWestLng = 99.67114584 - padding
    southEastLat = 13.91365211 - padding
    southEastLng = 99.67114584

    assert os.getenv("api_key") is not None, "Please set your API key in .env file"
    # Variables for API request (more info in README)
    zoom = 16
    picHeight = 640
    picWidth = 640
    scale = 2
    maptype = "satellite"
    print("START")
    # --- do not change variables below this point ---

    mapHeight = 256
    mapWidth = 256
    xScale = math.pow(2, zoom) / (picWidth / mapWidth)
    yScale = math.pow(2, zoom) / (picHeight / mapWidth)

    startLat = northWestLat
    startLng = northWestLng

    startCorners = getImageBounds(
        mapWidth, mapHeight, xScale, yScale, startLat, startLng
    )
    lngStep = startCorners[3] - startCorners[1]

    col = 0
    lat = startLat

    while lat >= southEastLat:
        lng = startLng
        row = 0

        while lng <= southEastLng:
            requestImage(picHeight, picWidth, zoom, scale, maptype, lat, lng, row, col)
            row = row + 1
            lng = lng + lngStep

        col = col - 1
        lat = lat + getLatStep(mapWidth, mapHeight, yScale, lat, lng)
    print("COMPLETE")


if __name__ == "__main__":
    main()
