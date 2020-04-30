package ai.rapids.cudf;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

class TestUtils {
  private static Random r = new Random();

  /**
   * A convenience method for generating a fixed set of Double values. This is by no means uniformly
   * distributed. i.e. some values have more probability of occurrence than others.
   *
   * @param seed seed to be used to generate values
   * @param size number of values to be generated
   */
  static Double[] getDoubles(final long seed, final int size, boolean nullsAllowed) {
    r.setSeed(seed);
    Double[] result = new Double[size];
    IntStream.range(0, size).forEach(index -> {
      switch (r.nextInt(4)) {
        case 0:
          result[index] = Double.MAX_VALUE * r.nextDouble();
          break;
        case 1:
          result[index] = Double.MIN_VALUE * r.nextDouble();
          break;
        case 2:
          result[index] = Double.NaN;
          break;
        case 3:
          result[index] = r.nextBoolean() ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
          break;
        default:
          result[index] = null;
      }
    });
    return result;
  }
}
