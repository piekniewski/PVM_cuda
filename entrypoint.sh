#!/usr/bin/env bash
set -e

## Run startup command
cd /pvm
pip3 install -e .

## Running passed command
if [[ "$1" ]]; then
	eval "$@"
fi

