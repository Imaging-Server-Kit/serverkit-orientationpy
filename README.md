![EPFL Center for Imaging logo](https://imaging.epfl.ch/resources/logo-for-gitlab.svg)
# serverkit-orientationpy

Implementation of a web server for [Orientationpy](https://gitlab.com/epfl-center-for-imaging/orientationpy).

Author: EPFL Center for Imaging

## Installing the algorithm server with `pip`

Install dependencies:

```
pip install -r requirements.txt
```

Run the server:

```
uvicorn main:app --host 0.0.0.0 --port 8000
```

The server will be running on http://localhost:8000.

## Endpoints

A documentation of the endpoints is automatically generated at http://localhost:8000/docs.

**GET endpoints**

- http://localhost:8000/: Listing of running algorithm servers.
- http://localhost:8000/version: Version of the `imaging-server-kit` package.
- http://localhost:8000/orientationpy/info: Web page displaying project metadata.
- http://localhost:8000/orientationpy/demo: Plotly Dash web demo app.
- http://localhost:8000/orientationpy/parameters: Json Schema representation of algorithm parameters.
- http://localhost:8000/orientationpy/sample_images: Bytes string representation of the sample images.

**POST endpoints**

- http://localhost:8000/orientationpy/process: Processing endpoint to run the algorithm.

## Running the server with `docker-compose`

To build the docker image and run a container for the algorithm server in a single command, use:

```
docker compose up
```

The server will be running on http://localhost:8000.

## Running the server with `docker`

Build the docker image:

```
docker build -t serverkit-orientationpy .
```

Run the server in a container:

```
docker run -it --rm -p 8000:8000 serverkit-orientationpy:latest
```

The server will be running on http://localhost:8000.

## Running unit tests

If you have implemented unit tests in the [tests/](./tests/) folder, you can run them using pytest:

```
pytest
```

if you are developing your server locally, or

```
docker run --rm serverkit-orientationpy:latest pytest
```

to run the tests in a docker container.

## [EPFL only] Publishing a docker image to [registry.rcp.epfl.ch](https://registry.rcp.epfl.ch/)

```
docker tag serverkit-orientationpy registry.rcp.epfl.ch/imaging-server-kit/serverkit-orientationpy
docker push registry.rcp.epfl.ch/imaging-server-kit/serverkit-orientationpy
```

## Sample images provenance

- `image1_from_OrientationJ.tif`: Example image from [OrientationJ](https://bigwww.epfl.ch/demo/orientationj/).
