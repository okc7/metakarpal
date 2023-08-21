#!/bin/bash

declare -A counts

for file in *.jpg.~*.jpg; do
    prefix="${file%%-*}"       # Extract prefix before the hyphen
    if [[ ! ${counts[$prefix]} ]]; then
        counts["$prefix"]=1    # Set count to 1 for the first instance
        new_name="${prefix}.jpg"
    else
        new_name="${prefix}-${counts[$prefix]}.jpg"  # Add count for subsequent instances
        counts["$prefix"]=$((counts["$prefix"]+1))   # Increment the count
    fi
    mv "$file" "$new_name"
done
