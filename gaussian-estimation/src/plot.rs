use anyhow::Result;
use ndarray::{Array1, Array2};
use plotters::prelude::*;

pub fn visualize_fit(x: &Array2<f64>, mu: &Array1<f64>, cov: &Array2<f64>, threshold: f64) -> Result<()> {
    let root = BitMapBackend::new("gaussian_outliers.png", (800, 600)).into_drawing_area();
    root.fill(&RGBColor(240, 240, 250))?;

    let x_range = (x.column(0).iter().fold(f64::INFINITY, |a, &b| a.min(b)) - 1.0, 
                   x.column(0).iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) + 1.0);
    let y_range = (x.column(1).iter().fold(f64::INFINITY, |a, &b| a.min(b)) - 1.0, 
                   x.column(1).iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) + 1.0);

    let mut chart = ChartBuilder::on(&root)
        .caption("Gaussian Distribution with Outliers", ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x_range.0..x_range.1, y_range.0..y_range.1)?;

    chart.configure_mesh()
        .x_desc("X")
        .y_desc("Y")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    let p = crate::multivariate_gaussian(x, mu, cov)?;
    let outliers = p.iter().map(|&pi| pi < threshold).collect::<Vec<_>>();

    // Plot normal points
    chart.draw_series(PointSeries::of_element(
        x.rows().into_iter().zip(outliers.iter())
            .filter(|(_, &is_outlier)| !is_outlier)
            .map(|(point, _)| (point[0], point[1])),
        2,
        ShapeStyle::from(&RGBColor(65, 105, 225)).filled(),
        &|coord, size, style| {
            EmptyElement::at(coord)
                + Circle::new((0, 0), size, style)
        },
    ))?
    .label("Normal Points")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RGBColor(65, 105, 225)));

    // Plot outliers
    chart.draw_series(PointSeries::of_element(
        x.rows().into_iter().zip(outliers.iter())
            .filter(|(_, &is_outlier)| is_outlier)
            .map(|(point, _)| (point[0], point[1])),
        3,
        ShapeStyle::from(&RGBColor(220, 20, 60)).filled(),
        &|coord, size, style| {
            EmptyElement::at(coord)
                + Circle::new((0, 0), size, style)
        },
    ))?
    .label("Outliers")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RGBColor(220, 20, 60)));

    // Draw mean point
    chart.draw_series(PointSeries::of_element(
        std::iter::once((mu[0], mu[1])),
        4,
        ShapeStyle::from(&RGBColor(50, 205, 50)).filled(),
        &|coord, size, style| {
            EmptyElement::at(coord)
                + Circle::new((0, 0), size, style)
        },
    ))?
    .label("Mean")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RGBColor(50, 205, 50)));

    // Add threshold information to the legend
    chart.draw_series(std::iter::once(EmptyElement::at((0.0, 0.0))))?
        .label(format!("Threshold: {:.6}", threshold));

    // Configure and draw the legend
    chart.configure_series_labels()
        .background_style(&RGBColor(240, 240, 250))
        .border_style(&RGBColor(100, 100, 100))
        .position(SeriesLabelPosition::UpperLeft)
        .draw()?;

    root.present()?;
    println!("Plot saved as gaussian_outliers.png");

    Ok(())
}