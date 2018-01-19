#!/usr/bin/env bash

for game in 'pong' 'CarRacing'; do
    mkdir ./${game}
    for set in 'images' 'actions'; do
        mkdir ./${game}/${set}
        chmod a+w ./${game}/${set}
        chmod a+r ./${game}/${set}
    done
    chmod a+w ${game}
    chmod a+r ${game}
done



