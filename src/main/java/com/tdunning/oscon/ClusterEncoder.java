package com.tdunning.oscon;

import com.google.common.base.CharMatcher;
import com.google.common.base.Charsets;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.io.Resources;
import com.tdunning.ch16.CategoryFeatureEncoder;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.ClusterClassifier;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

import java.io.IOException;
import java.util.List;

/**
 * Encode a donut data line using pre-defined k-means cluster centroids.
 */
public class ClusterEncoder {
  private static final Splitter onComma = Splitter.on(",").trimResults(CharMatcher.anyOf("\"\' "));
  private static final ImmutableList<String> X_Y = ImmutableList.of("x", "y");
  private static final int INTERNAL_VECTOR_SIZE = 5;

  private ClusterClassifier c;
  private DonutEncoder xyEncoder;
  private CategoryFeatureEncoder categoryEncoder;

  public ClusterEncoder(String name, Iterable<String> columns, String resourceName) throws IOException {
    List<Cluster> centroids = Lists.newArrayList();
    List<String> lines = Resources.readLines(Resources.getResource(resourceName), Charsets.UTF_8);
    DonutEncoder enc = new DonutEncoder(onComma.split(lines.get(0)), X_Y);

    for (String line : lines.subList(1, lines.size())) {
      Vector v = new RandomAccessSparseVector(INTERNAL_VECTOR_SIZE);
      enc.addToVector(onComma.split(line), v);
      Cluster c = new org.apache.mahout.clustering.kmeans.Cluster(v, 1, new EuclideanDistanceMeasure());
      centroids.add(c);
    }
    c = new ClusterClassifier(centroids);

    xyEncoder = new DonutEncoder(columns, X_Y);
    categoryEncoder = new CategoryFeatureEncoder(name);
  }

  public void addToVector(Iterable<String> values, Vector v) {
    int category = classify(values);
    categoryEncoder.addToVector(category, v);
  }

  public int classify(Iterable<String> values) {
    Vector xy = new RandomAccessSparseVector(INTERNAL_VECTOR_SIZE);
    xyEncoder.addToVector(values, xy);
    return c.classify(xy).maxValueIndex();
  }

}
