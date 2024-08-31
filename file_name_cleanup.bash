#!/bin/bash

# Iterate over all files with a specific pattern
for file in *.txt; do
  # Extract the part before the first underscore
  id="${file%%-*}"
  
  # Extract the part after the first underscore and replace underscores with hyphens
#   suffix="${file#*_}"
#   new_suffix="${suffix//_/-}"
  
  # Define the new file name
  new_name="${id}-discharge-inputs.txt"
  
  # Rename the file
  mv "$file" "$new_name"
  
  # Print a message indicating the rename was successful
  echo "Renamed $file to $new_name"
done
