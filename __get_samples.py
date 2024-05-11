import sys
import anndata as ad
import numpy as np

def get_samples(seed, file_path):
    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Load the .h5ad file
    adata = ad.read_h5ad(file_path)

    # Initialize an empty list to collect the indices of rows to keep
    indices_to_keep = []

    # Set the total maximum number of rows to collect
    total_max_rows = 300000
    current_row_count = 0

    # Get unique donor_ids and ensure it's a numpy array
    donor_ids = np.array(adata.obs['donor_id'].unique())

    # Shuffle the list of donor_ids for a random start
    np.random.shuffle(donor_ids)

    # Use a set to track added donor_ids
    added_donors = set()

    # Use a while loop to keep trying to add donor_ids until the condition is met
    while current_row_count < total_max_rows:
        for donor_id in donor_ids:
            # Skip donor_id if it has already been added
            if donor_id in added_donors:
                continue

            # Filter data for the current donor_id
            donor_data_indices = adata.obs.index[adata.obs['donor_id'] == donor_id]

            # Check if the total rows for this donor can be added without exceeding the limit
            if current_row_count + len(donor_data_indices) <= total_max_rows:
                # If adding this donor's rows won't surpass the limit, add them
                indices_to_keep.extend(donor_data_indices)
                current_row_count += len(donor_data_indices)

            # Mark this donor_id as attempted, regardless of success in adding rows
            added_donors.add(donor_id)

            # Stop the loop if the total maximum number of rows is already reached
            if current_row_count >= total_max_rows:
                break
        else:
            # If all donors have been tried and total_max rows is not reached, break the outer loop
            if len(added_donors) == len(donor_ids):
                break
            # Otherwise, continue looping through donor_ids
            continue
        break  # Exit the while loop if we've met the row count or added all possible donors

    # Create a new AnnData object with the filtered observations
    new_adata = adata[indices_to_keep]

    # Shuffle the new Anndata object's rows
    shuffled_indices = np.random.permutation(new_adata.shape[0])
    new_adata = new_adata[shuffled_indices]

    # Return the shuffled new AnnData object instead of writing it to a file
    return new_adata
