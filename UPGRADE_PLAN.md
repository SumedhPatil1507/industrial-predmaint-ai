# Premium Tier Upgrade Plan

## Current Issues
1. App requires file upload to do anything
2. Live data only in one page
3. No default dataset loaded
4. Model must be trained manually

## Premium Features to Add

### 1. Auto-Load Sample Data on Startup
- Generate 10,000 rows of synthetic data in-memory on first load
- Auto-train model on startup if not trained
- All pages work immediately without upload

### 2. Live Data Everywhere
- Dashboard shows live fleet metrics (auto-refreshing)
- EDA Explorer uses live-generated data by default
- Predict page has "Use Live Data" button
- All charts update in real-time

### 3. PDF Report Export
- Generate professional PDF report per asset
- Include: health score, TTF, sensor trends, recommendations
- Download button on every page

### 4. Comparison Mode
- Side-by-side comparison of 2-3 assets
- Overlay sensor trends
- Compare health scores

### 5. Alert Threshold Configuration
- User-configurable thresholds per sensor
- Visual threshold lines on all charts
- Alert when threshold breached

### 6. Historical Playback
- Scrub through historical data like a video
- See how degradation evolved
- Identify when failure started

### 7. What-If Simulator
- "What if vibration increases by 20%?"
- Show predicted impact on breakdown probability
- Sensitivity analysis charts

### 8. Mobile-Responsive Layout
- Collapsible sidebar
- Stacked charts on mobile
- Touch-friendly controls

## Implementation Priority
1. Auto-load sample data (30 min)
2. Live data everywhere (1 hour)
3. PDF export (45 min)
4. Comparison mode (1 hour)
5. Alert thresholds (30 min)
6. Historical playback (1 hour)
7. What-if simulator (45 min)
8. Mobile responsive (30 min)
