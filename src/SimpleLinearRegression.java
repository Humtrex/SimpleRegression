import java.util.List;

public class SimpleLinearRegression {
    private DataSet dataset;
    private Model model;

    public SimpleLinearRegression(DataSet dataset) {
        this.dataset = dataset;
        this.model = new Model();
    }

    public void calculateParameters() {
        double xSum = DiscreteMaths.sum(dataset.getX());
        double ySum = DiscreteMaths.sum(dataset.getY());
        double xySum = DiscreteMaths.sumOfProducts(dataset.getX(), dataset.getY());
        double xSquaredSum = DiscreteMaths.sumOfSquares(dataset.getX());
        int n = dataset.getX().size();

        double beta1 = (n * xySum - xSum * ySum) / (n * xSquaredSum - xSum * xSum);
        double beta0 = (ySum - beta1 * xSum) / n;

        model.setBeta_0(beta0);
        model.setBeta_1(beta1);
    }

    public Model getModel() {
        return model;
    }

    public double calculateRSquared(List<Double> x, List<Double> y) {
        double beta_0 = model.getBeta_0();
        double beta_1 = model.getBeta_1();

        double ssTot = 0.0;
        double ssRes = 0.0;
        double meanY = DiscreteMaths.sum(y) / y.size();

        for (int i = 0; i < x.size(); i++) {
            double predictedY = beta_0 + beta_1 * x.get(i);
            ssRes += Math.pow(y.get(i) - predictedY, 2);
            ssTot += Math.pow(y.get(i) - meanY, 2);
        }

        return 1 - (ssRes / ssTot);
    }

    // Método para hacer predicciones
    public double predict(double x) {
        return model.getBeta_0() + model.getBeta_1() * x;
    }
}

