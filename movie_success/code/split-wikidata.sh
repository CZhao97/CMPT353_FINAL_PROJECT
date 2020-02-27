#!/bin/sh

zcat $1 | split -C 2G --additional-suffix=.json -a 4 -d --filter="gzip -c > \$FILE.gz" - wd-

