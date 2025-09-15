import pandas as pd
import numpy as np

# Helper function: Min–Max normalization
def min_max_norm(series):
    return (series - series.min()) / (series.max() - series.min())

def compute_svi(df):
    """
    Social Vulnerability Index (SVI) factors:
      - Age: higher proportion of elderly and children increases vulnerability.
      - Socioeconomic: lower median income and lower educational attainment increase vulnerability.
      - Mobility: households with no vehicles are more vulnerable.
    
    We assign weights of 30% to age, 30% to socioeconomic factors, and 40% to mobility.
    """
    # Calculate age-related percentages
    df['elderly_pct'] = df['Households with one or more people 65 years and over'] / df['Number of households']
    df['child_pct']   = df['Households with one or more people under 18 years'] / df['Number of households']
    # Vehicle access vulnerability: higher percentage with no vehicles = higher vulnerability.
    df['no_vehicle_pct'] = df['Households with no vehicles'] / df['Number of households']

    # Normalize age and mobility percentages
    df['elderly_norm'] = min_max_norm(df['elderly_pct'])
    df['child_norm']   = min_max_norm(df['child_pct'])
    df['no_vehicle_norm'] = min_max_norm(df['no_vehicle_pct'])

    # Age factor: average of elderly and child normalized percentages.
    df['age_factor'] = (df['elderly_norm'] + df['child_norm']) / 2

    # Socioeconomic factors:
    # Income: lower median household income is more vulnerable.
    df['income_vuln'] = 1 - min_max_norm(df['Median household income (in US dollars)'])
    
    # Education: assume that a lower ratio of population with Bachelor's degrees implies higher vulnerability.
    df['education_pct'] = df["Population with Bachelor's degree or higher"] / df['Population']
    df['education_vuln'] = 1 - min_max_norm(df['education_pct'])
    
    # Average the two for the socioeconomic factor.
    df['socioeconomic_factor'] = (df['income_vuln'] + df['education_vuln']) / 2

    # Mobility factor: use the no-vehicle normalized percentage.
    df['mobility_factor'] = df['no_vehicle_norm']

    # Combine the three factors with the specified weights.
    df['SVI'] = 0.3 * df['age_factor'] + 0.3 * df['socioeconomic_factor'] + 0.4 * df['mobility_factor']
    # Scale to 0–10
    df['SVI_scaled'] = df['SVI'] * 10
    return df

def compute_ivi(df):
    """
    Infrastructure Vulnerability Index (IVI) factors:
      - Housing: Proportion of older homes (e.g., built before 1970) is a proxy for outdated infrastructure.
      - Open Space: Higher proportions of developed, open space can amplify urban heat island effects.
    
    Both are normalized and weighted equally.
    """
    # Calculate older housing proportion.
    # We assume older homes are those built in 1950 or earlier plus those built between 1950 and 1969.
    df['older_homes'] = df['Homes built 1950 or earlier'] + df['Homes built 1950 to 1969']
    # Total dwellings is the sum of all dwelling types.
    df['total_dwellings'] = df['Detached whole house'] + df['Townhouse'] + df['Apartments'] + df['Mobile Homes/Other']
    df['older_home_pct'] = df['older_homes'] / df['total_dwellings']
    
    # Normalize the older home percentage.
    df['older_home_norm'] = min_max_norm(df['older_home_pct'])
    
    # Normalize the proportion of developed, open space.
    df['open_space_norm'] = min_max_norm(df['Proportion of developed, open space in neighborhood¹'])
    
    # Combine with equal weighting.
    df['IVI'] = 0.5 * df['older_home_norm'] + 0.5 * df['open_space_norm']
    df['IVI_scaled'] = df['IVI'] * 10
    return df

def compute_mai(df):
    """
    Mobility and Access Index (MAI):
      - Based on the primary mode of transportation to work.
      - A higher proportion of residents relying on walking or public transit (in a context of limited service during emergencies) indicates higher vulnerability.
    
    The proportion of non-driving commuters is normalized and used directly.
    """
    df['non_driving_pct'] = df['Primary mode of transportation to work (persons aged 16 years+): walking or public transit'] / \
                              df['Population age 16+ years old who work']
    df['non_driving_norm'] = min_max_norm(df['non_driving_pct'])
    df['MAI'] = df['non_driving_norm']
    df['MAI_scaled'] = df['MAI'] * 10
    return df

def compute_composite_score(df):
    """
    Composite Vulnerability Score (CVS):
      - Combines SVI (40%), IVI (30%), and MAI (30%) into a single metric.
    """
    df['CVS'] = 0.4 * df['SVI'] + 0.3 * df['IVI'] + 0.3 * df['MAI']
    df['CVS_scaled'] = df['CVS'] * 10
    return df

def main():
    # Load data from a CSV file. The CSV file should contain columns matching those used below.
    # For example, your CSV might be named "memphis_neighborhoods.csv".
    df = pd.read_csv('memphis_neighborhoods.csv')
    
    # Compute each index step by step.
    df = compute_svi(df)
    df = compute_ivi(df)
    df = compute_mai(df)
    df = compute_composite_score(df)
    
    # Display selected columns including the computed scores.
    score_columns = ['Neighborhood', 'ZIP code', 'SVI_scaled', 'IVI_scaled', 'MAI_scaled', 'CVS_scaled']
    print(df[score_columns])
    
    # Optionally, write the results to a new CSV.
    df.to_csv('memphis_vulnerability_scores.csv', index=False)

if __name__ == '__main__':
    main()
