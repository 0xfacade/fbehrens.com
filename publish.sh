#!/bin/bash

pushd themes/minimo-fb
node_modules/.bin/webpack
popd

rm -r public
hugo
rsync -avz --delete public/ fbehrens.com:/var/www/blog
