type Weight = Vec<f64>;
type Data = Vec<f64>;

fn main() {
    let w: Weight = vec![1., 2., 3., 4.];
    let x: Data = vec![1., 2., 3.];
    println!("W = {:?}", w);
    println!("x = {:?}", x);

    let y: f64 = weight_sum(w, x);
    println!("y = W^T*x = {}", y);

    let s: f64 = sigmoid(y);
    println!("y = sigmoid(o) = {}", s);
}

// y = W^T*x = sum_{i=1}^d w_i * x_i + w_0
fn weight_sum(w: Weight, x: Data) -> f64 {
    assert_eq!(w.len(), x.len() + 1);

    let mut sum = w[0]; // w_0
    
    for i in 1..w.len() {
        sum = sum + w[i] * x[i - 1];
    }
    sum
}

fn sigmoid(o: f64) -> f64 {
    1. / (1. + ((-1.) * o).exp())
}
