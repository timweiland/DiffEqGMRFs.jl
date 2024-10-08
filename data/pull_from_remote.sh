source .env

rsync -chavzP --stats $REMOTEDIR $LOCALDIR
