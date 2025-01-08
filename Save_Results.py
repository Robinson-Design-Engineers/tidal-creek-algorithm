import numpy as np #type:ignore
import matplotlib.pyplot as plt #type:ignore
import pandas as pd #type:ignore

# Step 10 function:
# Create and save result tables
def process_creek_morphometry(ANGLEORDER, ID, SEGMENTS, SINUOUSLENGTH, SINUOSITY, WIDTH, 
                            DEPTH, AREA, STRAIGHTDIST, name, Cth, LZth, HZth, 
                            Ctharea, LZtharea, HZtharea):
    
    # Convert inputs to numpy arrays if they aren't already
    ID = np.asarray(ID).reshape(-1)
    SEGMENTS = np.asarray(SEGMENTS).reshape(-1)
    
    # Process ANGLEORDER
    A2 = ANGLEORDER.copy()
    A2[ANGLEORDER == 0] = np.nan
    
    if A2.shape[0] < len(ID):
        nanrow = np.full((1, A2.shape[1]), np.nan)
        A2 = np.vstack([A2, nanrow])
        ANGLEORDER = np.vstack([ANGLEORDER, nanrow])
    
    # Calculate mean column (excluding NaN values)
    with np.errstate(invalid='ignore'):
        meancol = np.nanmean(A2, axis=1, keepdims=True)
        ANGLEORDER = np.hstack([ANGLEORDER, meancol])
    
    # Replace zeros with NaN for various measurements
    for arr in [AREA, SINUOUSLENGTH, STRAIGHTDIST, WIDTH, DEPTH]:
        arr = np.asarray(arr)
        arr[arr <= 0] = np.nan
    
    # Shift content up for WIDTH, DEPTH, and AREA
    # Move content from rows 1,2,3 to rows 0,1,2
    WIDTH = WIDTH[1:]  # Remove first row
    DEPTH = DEPTH[1:]  # Remove first row
    AREA = AREA[1:]   # Remove first row
    
    # Ensure all arrays have the same number of rows as ID
    n_rows = len(ID)
    
    def pad_array(arr, target_rows):
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[0] < target_rows:
            padding = np.full((target_rows - arr.shape[0], arr.shape[1]), np.nan)
            return np.vstack([arr, padding])
        return arr
   
    SINUOUSLENGTH = pad_array(SINUOUSLENGTH, n_rows)
    STRAIGHTDIST = pad_array(STRAIGHTDIST, n_rows)
    SINUOSITY = pad_array(SINUOSITY, n_rows)
    WIDTH = pad_array(WIDTH, n_rows)
    DEPTH = pad_array(DEPTH, n_rows)
    AREA = pad_array(AREA, n_rows)

    # prepare for mean calculations
    SEGMENTS_reshaped = SEGMENTS.reshape(-1, 1)
    safe_segments = np.where(SEGMENTS_reshaped == 0, np.nan, SEGMENTS_reshaped) # Avoid division by zero by replacing zero SEGMENTS with NaN

    # Create summary data with error handling
    with np.errstate(invalid='ignore'):
        summary_data = np.column_stack([
            ID.reshape(-1, 1),
            SEGMENTS.reshape(-1, 1),
            np.nansum(SINUOUSLENGTH, axis=1, keepdims=True),
            (np.nansum(SINUOUSLENGTH, axis=1, keepdims=True) / safe_segments),
            (np.nansum(SINUOSITY, axis=1, keepdims=True) / safe_segments),
            (np.nansum(WIDTH, axis=1, keepdims=True) / safe_segments),
            (np.nansum(DEPTH, axis=1, keepdims=True) / safe_segments),
            (np.nansum(AREA, axis=1, keepdims=True) / safe_segments),
            np.nansum(STRAIGHTDIST, axis=1, keepdims=True),
            (np.nansum(STRAIGHTDIST, axis=1, keepdims=True) / safe_segments),
        ])
    
    summary_columns = [
        'RS order', '# segments', 'Total sinuous length', 'Mean sinuous length',
        'Mean sinuosity', 'Mean channel width', 'Mean channel depth',
        'Mean cross-sectional area', 'Total channel length', 'Mean channel length'
    ]
    
    # Create DataFrame for summary table
    SUMMARY_df = pd.DataFrame(summary_data, columns=summary_columns)
    # Reverse the values of the 'RS order' column
    SUMMARY_df['RS order'] = SUMMARY_df['RS order'][::-1].values
    
    # Create detailed measurement tables
    with np.errstate(invalid='ignore'):
        SINUOUSLENGTH = np.column_stack([
            ID.reshape(-1, 1),
            SEGMENTS.reshape(-1, 1),
            np.nansum(SINUOUSLENGTH, axis=1, keepdims=True),
            np.nanmean(SINUOUSLENGTH, axis=1, keepdims=True),
            SINUOUSLENGTH
        ])
        
        STRAIGHTDIST = np.column_stack([
            ID.reshape(-1, 1),
            SEGMENTS.reshape(-1, 1),
            np.nansum(STRAIGHTDIST, axis=1, keepdims=True),
            np.nanmean(STRAIGHTDIST, axis=1, keepdims=True),
            STRAIGHTDIST
        ])
        
        SINUOSITY = np.column_stack([
            ID.reshape(-1, 1),
            SEGMENTS.reshape(-1, 1),
            np.nanmean(SINUOSITY, axis=1, keepdims=True),
            SINUOSITY
        ])
        
        DEPTH = np.column_stack([DEPTH, np.nanmean(DEPTH, axis=1, keepdims=True)])
        WIDTH = np.column_stack([WIDTH, np.nanmean(WIDTH, axis=1, keepdims=True)])
        AREA = np.column_stack([AREA, np.nanmean(AREA, axis=1, keepdims=True)])
    
    # Create and display summary table
    plt.figure(figsize=(13, 2))
    plt.axis('tight')
    plt.axis('off')
    table = plt.table(cellText=SUMMARY_df.values,
                     colLabels=SUMMARY_df.columns,
                     cellLoc='center',
                     loc='center')
    
    # Save summary table to Excel
    output_path = f'Outputs/{name}_CreekMorphometry.xlsx'
    SUMMARY_df.to_excel(output_path, index=False)
    
    # Create and display thresholds table
    thresh_data = np.array([
        [Cth, LZth, HZth],
        [Ctharea, LZtharea, HZtharea]
    ])
    
    thresh_columns = ['Slope threshold Sth', 
                     'Low elevation threshold LZth', 
                     'High elevation threshold HZth']
    thresh_index = ['Value', 'Area']
    
    THRESHOLD_df = pd.DataFrame(thresh_data, 
                           columns=thresh_columns,
                           index=thresh_index)
    
    plt.figure(figsize=(8, 1.5))
    plt.axis('tight')
    plt.axis('off')
    thresh_table = plt.table(cellText=THRESHOLD_df.values,
                           colLabels=THRESHOLD_df.columns,
                           rowLabels=THRESHOLD_df.index,
                           cellLoc='center',
                           loc='center')
    
    plt.show()
    
    return (SUMMARY_df, THRESHOLD_df, SINUOUSLENGTH, STRAIGHTDIST, 
            SINUOSITY, DEPTH, WIDTH, AREA)

def process_creek_morphometry_diagnostic(ANGLEORDER, ID, SEGMENTS, SINUOUSLENGTH, SINUOSITY, WIDTH, 
                            DEPTH, AREA, STRAIGHTDIST, name, Cth, LZth, HZth, 
                            Ctharea, LZtharea, HZtharea):
    
    print("\nOriginal shapes:")
    print(f"ID shape before reshape: {np.asarray(ID).shape}")
    print(f"SEGMENTS shape before reshape: {np.asarray(SEGMENTS).shape}")
    
    # Convert inputs to numpy arrays if they aren't already
    ID = np.asarray(ID).reshape(-1)[::-1]
    SEGMENTS = np.asarray(SEGMENTS).reshape(-1)
    
    print("\nAfter reshape:")
    print(f"ID shape: {ID.shape}")
    print(f"SEGMENTS shape: {SEGMENTS.shape}")

    
    # Process ANGLEORDER
    A2 = ANGLEORDER.copy()
    A2[ANGLEORDER == 0] = np.nan
    
    if A2.shape[0] < len(ID):
        nanrow = np.full((1, A2.shape[1]), np.nan)
        A2 = np.vstack([A2, nanrow])
        ANGLEORDER = np.vstack([ANGLEORDER, nanrow])
    
    # Calculate mean column (excluding NaN values)
    with np.errstate(invalid='ignore'):
        meancol = np.nanmean(A2, axis=1, keepdims=True)
        ANGLEORDER = np.hstack([ANGLEORDER, meancol])
    
    # Replace zeros with NaN for various measurements
    for arr in [AREA, SINUOUSLENGTH, STRAIGHTDIST, WIDTH, DEPTH]:
        arr = np.asarray(arr)
        arr[arr <= 0] = np.nan
    
    # Shift content up for WIDTH, DEPTH, and AREA
    print("\nBefore shifting:")
    print("WIDTH rows:", WIDTH[:8] if WIDTH is not None else "None")
    print("DEPTH rows:", DEPTH[:8] if DEPTH is not None else "None")
    print("AREA rows:", AREA[:8] if AREA is not None else "None")
    
    # Move content from rows 1,2,3 to rows 0,1,2
    # Delete the first row
    WIDTH = np.delete(WIDTH, 0, axis=0)
    DEPTH = np.delete(DEPTH, 0, axis=0)
    AREA = np.delete(AREA, 0, axis=0)
    
    print("\nAfter shifting:")
    print("WIDTH rows:", WIDTH[:8] if WIDTH is not None else "None")
    print("DEPTH rows:", DEPTH[:8] if DEPTH is not None else "None")
    print("AREA rows:", AREA[:8] if AREA is not None else "None")
    
    # Ensure all arrays have the same number of rows as ID
    n_rows = len(ID)
    
    def pad_array(arr, target_rows):
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[0] < target_rows:
            padding = np.full((target_rows - arr.shape[0], arr.shape[1]), np.nan)
            return np.vstack([arr, padding])
        return arr
    
    print(f"Target number of rows: {n_rows}")
    
    print("\nOriginal array shapes:")
    print(f"SINUOUSLENGTH shape: {np.asarray(SINUOUSLENGTH).shape}")
    print(f"STRAIGHTDIST shape: {np.asarray(STRAIGHTDIST).shape}")
    print(f"SINUOSITY shape: {np.asarray(SINUOSITY).shape}")
    print(f"WIDTH shape: {np.asarray(WIDTH).shape}")
    print(f"DEPTH shape: {np.asarray(DEPTH).shape}")
    print(f"AREA shape: {np.asarray(AREA).shape}")
    
    SINUOUSLENGTH = pad_array(SINUOUSLENGTH, n_rows)
    STRAIGHTDIST = pad_array(STRAIGHTDIST, n_rows)
    SINUOSITY = pad_array(SINUOSITY, n_rows)
    WIDTH = pad_array(WIDTH, n_rows)
    DEPTH = pad_array(DEPTH, n_rows)
    AREA = pad_array(AREA, n_rows)
    
    print("\nPadded array shapes:")
    print(f"SINUOUSLENGTH shape: {SINUOUSLENGTH.shape}")
    print(f"STRAIGHTDIST shape: {STRAIGHTDIST.shape}")
    print(f"SINUOSITY shape: {SINUOSITY.shape}")
    print(f"WIDTH shape: {WIDTH.shape}")
    print(f"DEPTH shape: {DEPTH.shape}")
    print(f"AREA shape: {AREA.shape}")
    
    # Create summary data with error handling
    with np.errstate(invalid='ignore'):
        summary_data = np.column_stack([
            ID.reshape(-1, 1),
            SEGMENTS.reshape(-1, 1),
            np.nansum(SINUOUSLENGTH, axis=1, keepdims=True),
            np.nanmean(SINUOUSLENGTH, axis=1, keepdims=True),
            np.nanmean(SINUOSITY, axis=1, keepdims=True),
            np.nanmean(WIDTH, axis=1, keepdims=True),
            np.nanmean(DEPTH, axis=1, keepdims=True),
            np.nanmean(AREA, axis=1, keepdims=True),
            np.nansum(STRAIGHTDIST, axis=1, keepdims=True),
            np.nanmean(STRAIGHTDIST, axis=1, keepdims=True)
        ])
    
    summary_columns = [
        'RS order', '# segments', 'Total sinuous length', 'Mean sinuous length',
        'Mean sinuosity', 'Mean channel width', 'Mean channel depth',
        'Mean cross-sectional area', 'Total channel length', 'Mean channel length'
    ]
    
    # Create DataFrame for summary table
    SUMMARY_df = pd.DataFrame(summary_data, columns=summary_columns)
    # Reverse the values of the 'RS order' column
    SUMMARY_df['RS order'] = SUMMARY_df['RS order'][::-1].values
    
    # Create detailed measurement tables
    with np.errstate(invalid='ignore'):
        SINUOUSLENGTH = np.column_stack([
            ID.reshape(-1, 1),
            SEGMENTS.reshape(-1, 1),
            np.nansum(SINUOUSLENGTH, axis=1, keepdims=True),
            np.nanmean(SINUOUSLENGTH, axis=1, keepdims=True),
            SINUOUSLENGTH
        ])
        
        STRAIGHTDIST = np.column_stack([
            ID.reshape(-1, 1),
            SEGMENTS.reshape(-1, 1),
            np.nansum(STRAIGHTDIST, axis=1, keepdims=True),
            np.nanmean(STRAIGHTDIST, axis=1, keepdims=True),
            STRAIGHTDIST
        ])
        
        SINUOSITY = np.column_stack([
            ID.reshape(-1, 1),
            SEGMENTS.reshape(-1, 1),
            np.nanmean(SINUOSITY, axis=1, keepdims=True),
            SINUOSITY
        ])
        
        DEPTH = np.column_stack([DEPTH, np.nanmean(DEPTH, axis=1, keepdims=True)])
        WIDTH = np.column_stack([WIDTH, np.nanmean(WIDTH, axis=1, keepdims=True)])
        AREA = np.column_stack([AREA, np.nanmean(AREA, axis=1, keepdims=True)])
    
    # Create and display summary table
    plt.figure(figsize=(13, 2))
    plt.axis('tight')
    plt.axis('off')
    table = plt.table(cellText=SUMMARY_df.values,
                     colLabels=SUMMARY_df.columns,
                     cellLoc='center',
                     loc='center')
    
    # Save summary table to Excel
    output_path = f'Outputs/{name}_CreekMorphometry.xlsx'
    SUMMARY_df.to_excel(output_path, index=False)
    
    # Create and display thresholds table
    thresh_data = np.array([
        [Cth, LZth, HZth],
        [Ctharea, LZtharea, HZtharea]
    ])
    
    thresh_columns = ['Slope threshold Sth', 
                     'Low elevation threshold LZth', 
                     'High elevation threshold HZth']
    thresh_index = ['Value', 'Area']
    
    THRESHOLD_df = pd.DataFrame(thresh_data, 
                           columns=thresh_columns,
                           index=thresh_index)
    
    plt.figure(figsize=(8, 1.5))
    plt.axis('tight')
    plt.axis('off')
    thresh_table = plt.table(cellText=THRESHOLD_df.values,
                           colLabels=THRESHOLD_df.columns,
                           rowLabels=THRESHOLD_df.index,
                           cellLoc='center',
                           loc='center')
    
    plt.show()
    
    return (SUMMARY_df, THRESHOLD_df, SINUOUSLENGTH, STRAIGHTDIST, 
            SINUOSITY, DEPTH, WIDTH, AREA)