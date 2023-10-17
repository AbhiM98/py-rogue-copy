call poetry env use "C:\Python38\python.exe"
call poetry cache clear pypi --all
call poetry run pip cache purge
call poetry install
call poetry run pip install pipwin 
call poetry run pipwin refresh
call poetry run pipwin install gdal
call poetry run pipwin install fiona 
call poetry run pip install geopandas
call poetry install 
call poetry run pip install -e .
