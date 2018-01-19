#!/usr/bin/env bash

for game in 'pong'; do
    mkdir ./${game}
    touch ./${game}/states_list.txt
    chmod a+w ./${game}/states_list.txt
    chmod a+r ./${game}/states_list.txt
    for set in 'images' 'actions'; do
        mkdir ./${game}/${set}
        chmod a+w ./${game}/${set}
        chmod a+r ./${game}/${set}
    done
done



