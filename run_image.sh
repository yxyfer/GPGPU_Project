cd build/


if [ $# -eq 0 ]
    then
        ./main_gpu ../images/pinguin_video/thumb*.jpg 
else
    ./main_gpu $1
fi

cd ../
