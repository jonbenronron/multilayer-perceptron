type Weight = f64;
type Weights = Vec<Weight>;
type Data = Vec<f64>;
type Output = f64;

fn main() {
    let weights: Weights = vec![1., 2., 3., 4.];
    let data: Data = vec![1., 2., 3.];
    println!("W = {:?}", &weights);
    println!("x = {:?}", &data);

    let err: f64 = linear_regression_error(&weights, &data, &0.);
    println!("E: {}", err);

    let y: Output = weight_sum(&weights, &data);
    let d_o: Output = y - err;

    println!("Sigmoid: {}", sigmoid(&y));

    for x in data.iter() {
        let dw: Weight = delta_w(&1., &d_o, &y, &x);
        println!("Delta w = {}", &dw);
    }
}

// y = W^T*x = sum_{i=1}^d w_i * x_i + w_0
fn weight_sum(w: &Weights, x: &Data) -> Output {
    assert_eq!(w.len(), x.len() + 1);

    let mut sum = w[0]; // w_0
    
    for i in 1..w.len() {
        sum = sum + w[i] * x[i - 1];
    }
    sum
}

// y = sigmoid(o) = 1 / (1 + exp(-W^T*x))
fn sigmoid(o: &f64) -> f64 {
    1. / (1. + ((-1.) * o).exp())
}

// LMS rule
fn delta_w(
    learning_factor: &f64,
    desired_output: &f64,
    actual_output: &f64,
    input: &f64
) -> Weight {
    learning_factor * (desired_output - actual_output) * input
}

// Regression (Linear output) error
fn linear_regression_error(w: &Weights, x: &Data, desired_output: &Output) -> f64 {
    0.5 * (desired_output - weight_sum(w, x)).powi(2)
}
