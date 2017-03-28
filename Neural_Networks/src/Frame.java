import java.util.ArrayList;

/**
 * Created by Fr0sk on 28/03/2017.
 */
public class Frame {
    private double timestamp;
    private ArrayList<Double> x, y, z;
    private boolean positive;

    public Frame(double timestamp) {
        this.timestamp = timestamp;
        x = new ArrayList<>();
        y = new ArrayList<>();
        z = new ArrayList<>();
    }

    public Frame(double timestamp, ArrayList<Double> x, ArrayList<Double> y, ArrayList<Double> z, boolean positive) {
        this.timestamp = timestamp;
        this.x = x;
        this.y = y;
        this.z = z;
        this.positive = positive;
    }

    public double getTimestamp() {
        return timestamp;
    }

    public ArrayList<Double> getX() {
        return x;
    }

    public ArrayList<Double> getY() {
        return y;
    }

    public ArrayList<Double> getZ() {
        return z;
    }

    public boolean isPositive() {
        return positive;
    }

    public void addToX(Double d) {
        x.add(d);
    }

    public void addToY(Double d) {
        y.add(d);
    }

    public void addToZ(Double d) {
        z.add(d);
    }
}
