#!/usr/bin/env bash

for game in 'pong'; do
    mkdir ./${game}
    for set in 'images' 'actions'; do
        mkdir ./${game}/${set}
        chmod a+w ./${game}/${set}
        chmod a+r ./${game}/${set}
    done
done



