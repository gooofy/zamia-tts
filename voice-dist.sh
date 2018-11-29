#!/bin/bash

if [ $# != 2 ] ; then

    echo "usage: $0 <voice> <epoch>"
    echo

    exit 1

fi

VOICE=$1
EPOCH=$2

datum=`date +%Y%m%d`
REVISION="r$datum"
ARCNAME="voice-${VOICE}-${REVISION}"
DISTDIR=data/dist/tts

echo "${ARCNAME}..."

rm -rf "$DISTDIR/$ARCNAME"
mkdir -p "$DISTDIR/$ARCNAME"

cp data/dst/tts/voices/${VOICE}/cp/cp${EPOCH}-*.data-00000-of-00001 "$DISTDIR/$ARCNAME/model.data-00000-of-00001"
cp data/dst/tts/voices/${VOICE}/cp/cp${EPOCH}-*.index               "$DISTDIR/$ARCNAME/model.index"
cp data/dst/tts/voices/${VOICE}/cp/cp${EPOCH}-*.meta                "$DISTDIR/$ARCNAME/model.meta"
cp data/dst/tts/voices/${VOICE}/hparams.json                        "$DISTDIR/$ARCNAME/hparams.json"

cp README.adoc LICENSE AUTHORS                                      "$DISTDIR/$ARCNAME/"

pushd $DISTDIR
tar cfv "$ARCNAME.tar" $ARCNAME
xz -v -8 -T 12 "$ARCNAME.tar"
popd

rm -r "$DISTDIR/$ARCNAME"

#
# upload
#

echo rsync -avPz --delete --bwlimit=256 data/dist/tts/ goofy:/var/www/html/zamia-tts/

