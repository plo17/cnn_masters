import mysql.connector

def training_results(model_name, epoch, train_acc, val_acc, train_loss, val_loss):
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="magda2000",
        database="MLflowDB"
    )
    cursor = conn.cursor()

    sql = """INSERT INTO models (model_name, epoch, train_accuracy, val_accuracy, train_loss, val_loss)
             VALUES (%s, %s, %s, %s, %s, %s)"""

    cursor.execute(sql, (model_name, epoch, train_acc, val_acc, train_loss, val_loss))
    conn.commit()
    conn.close()



def model_metrics(model_name, accuracy, precision, recall, f1):
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="magda2000",
        database="MLflowDB"
    )
    cursor = conn.cursor()

    sql = """INSERT INTO model_metrics (model_name, accuracy, precision, recall, f1_score)
             VALUES (%s, %s, %s, %s, %s)"""

    cursor.execute(sql, (model_name, accuracy, precision, recall, f1))
    conn.commit()
    conn.close()
    print("Metryki modelu zapisane w bazie!")
