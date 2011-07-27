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

  private ClusterClassifier clustering;
  private DonutEncoder xyEncoder;
  private CategoryFeatureEncoder categoryEncoder;

  /**
   * Defines a cluster encoder.
   * @param name               The name of the cluster variable in the final encoded form
   * @param columns            The names of the columns in the actual data
   * @param resourceName       The name of the resource that contains the centroids
   * @throws IOException       If we can't read the centroids for some reason
   */
  public ClusterEncoder(String name, Iterable<String> columns, String resourceName) throws IOException {
    List<Cluster> centroids = Lists.newArrayList();

    // read centroids file
    List<String> lines = Resources.readLines(Resources.getResource(resourceName), Charsets.UTF_8);

    // this encoder is used to encode data from the centroids file
    DonutEncoder enc = new DonutEncoder(onComma.split(lines.get(0)), X_Y);

    // now we encode each centroid line
    for (String line : lines.subList(1, lines.size())) {
      Vector v = new RandomAccessSparseVector(INTERNAL_VECTOR_SIZE);
      enc.addToVector(onComma.split(line), v);

      // with that centroid, we make a cluster and record it
      Cluster c = new org.apache.mahout.clustering.kmeans.Cluster(v, 1, new EuclideanDistanceMeasure());
      centroids.add(c);
    }

    // the list of centroids gives us a cluster classifier
    clustering = new ClusterClassifier(centroids);

    // now we need an encoder that will read the production data
    xyEncoder = new DonutEncoder(columns, X_Y);

    // and an encoder that can record the clustering results
    categoryEncoder = new CategoryFeatureEncoder(name);
  }

  /**
   * Classifies incoming data into a cluster and encodes the cluster number as a categorical value.
   * @param values   Fields of data
   * @param v        Vector to encode to
   */
  public void addToVector(Iterable<String> values, Vector v) {
    int category = classify(values);
    categoryEncoder.addToVector(category, v);
  }

  /**
   * Extract coordinates of the cluster and ask the clustering which class is best
   * @param values    Fields of data to cluster
   * @return   An integer indicating which cluster was the best fit
   */
  public int classify(Iterable<String> values) {
    Vector xy = new RandomAccessSparseVector(INTERNAL_VECTOR_SIZE);
    xyEncoder.addToVector(values, xy);
    return clustering.classify(xy).maxValueIndex();
  }
}
