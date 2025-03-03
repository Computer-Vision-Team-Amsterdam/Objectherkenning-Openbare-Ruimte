from pyspark.sql import functions as F


def get_score_expr():
    """
    Returns a column expression for scoring detections.
    For now, the same scoring function is applied for all object classes.
    """
    return (
        F.when(
            F.col("object_class") == 2,
            F.when(
                (F.col("closest_permit_distance") >= 40)
                & (F.col("closest_bridge_distance") < 25),
                1 + F.greatest((25 - F.col("closest_bridge_distance")) / 25, F.lit(0)),
            )
            .when(
                (F.col("closest_permit_distance") >= 40)
                & (F.col("closest_bridge_distance") >= 25),
                F.least(F.lit(1.0), F.col("closest_permit_distance") / 100),
            )
            .otherwise(0),
        )
        .when(
            F.col("object_class") == 3,
            F.when(
                (F.col("closest_permit_distance") >= 40)
                & (F.col("closest_bridge_distance") < 25),
                1 + F.greatest((25 - F.col("closest_bridge_distance")) / 25, F.lit(0)),
            )
            .when(
                (F.col("closest_permit_distance") >= 40)
                & (F.col("closest_bridge_distance") >= 25),
                F.least(F.lit(1.0), F.col("closest_permit_distance") / 100),
            )
            .otherwise(0),
        )
        .when(
            F.col("object_class") == 4,
            F.when(
                (F.col("closest_permit_distance") >= 40)
                & (F.col("closest_bridge_distance") < 25),
                1 + F.greatest((25 - F.col("closest_bridge_distance")) / 25, F.lit(0)),
            )
            .when(
                (F.col("closest_permit_distance") >= 40)
                & (F.col("closest_bridge_distance") >= 25),
                F.least(F.lit(1.0), F.col("closest_permit_distance") / 100),
            )
            .otherwise(0),
        )
        .otherwise(0)
    )
