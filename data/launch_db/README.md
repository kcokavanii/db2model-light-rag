# Самостоятельное развёртывание PostgreSQL

Учебный PostgreSQL-сервер доступен только участникам с выданной личной учётной
записью. Пользователю без такого доступа необходимо самостоятельно развернуть
PostgreSQL и импортировать нужные базы. Эта папка содержит вспомогательные
скрипты миграции, но не готовый `docker-compose.yml` и не сами datasets.

## Launching docker-compose

To host the PostgreSQL database, you'll first need to launch the `docker-compose.yml` file.

For safety purposes, the `docker-compose.yml` has to be obtained from other developers working on this benchmark.

First, set all the necessary `env` variables for a PostgreSQL db (use an `.env` file for these for conveience and security):
`POSTGRES_USER`, `POSTGRES_PASSWORD` and `POSTGRES_DB`. Also choose an open port instead of 5444 and set it in the docker file.
These are your "admin" user's settings, you should save these.

**Make sure your password is a real secure password and not something like `admin` or `123`!!!**

Now you can launch the db with something like this (probably in a `tmux` session, or with a detached container (add `-d` to the command)):
```
docker-compose up --build
```

## Postgres read-only user

In order for students to not break the database accidentally, they should receive a special **"read-only" user** instead of your **"admin" user**. Create a username and a strong password for students to use.

After that, run this command to enter the database console (replace `container_name` with the name of the real docker container (check with `docker ps`) and `admin_user_name` with what you set as `POSTGRES_USER`):
```bash
docker exec -it container_name psql -U admin_user_name -d postgres
```

In the console type the following commands (replace all mentions of `readonly_user` and `VeryStrongPassword` respectively with the username and password you just came up with). Of course, you could just keep the `readonly_user` username and only create a new password.
```bash
CREATE USER readonly_user WITH PASSWORD 'VeryStrongPassword';
GRANT pg_read_all_data TO readonly_user;
ALTER ROLE readonly_user SET default_transaction_read_only = on;
```

## Uploading datasets

There are two benchmarks right now - BIRD and Ambrosia.
You will need to install both datasets. For simplicity, upload the BIRD dataset first, instructions below.

### BIRD

You can install their `dev` dataset from their [website](https://bird-bench.github.io/) with something like this:
```bash
wget https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip
```
or this:
```bash
curl -L https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip -o dev.zip
```

Then, unpack it:
```bash
unzip dev.zip
```
It will have another zip file inside - `/dev_some_number/dev_databases.zip`, unpack it too:
```bash
unzip dev_databases.zip
cd dev_databases
```

One of the databases doesn't work - delete the `european_football_2` db. We don't include it in the datasets.
```bash
rmdir -rf ./european_football_2
```

Next, there is a bash script `migrate.sh` right in **this** directory, which can migrate all of these SQLite databases to PostgreSQL and upload them into out database.

Put the `migrate.sh` file there (you could just create a new file and copy the code there). Change the `PG_CONTAINER` variable to the real docker container name, also change `admin_username` and `admin_password` to the real username and password of your postgres database. Also check you're accessing the right port in the URI.

Finally, you can run the migration:
```bash
./migrate.sh
```
In case one of the db's breaks, you can just delete it, just be mindful of the question datasets (both `dev` and `train`)

> Ограничение текущего служебного скрипта: параметры подключения в
> `migrate.sh` задаются вручную и не загружаются из переменных окружения.

### AMBROSIA

Ambrosia has a different policy, so you can't just install it with `curl` or `wget`, you need to go to their [website](https://ambrosia-benchmark.github.io/) and follow their instructions on downloading the dataset. (at the moment of writing it's "follow a link and enter a given password")

Once you install the zip file, unpack it:
```bash
unzip data.zip
cd data
```
Put the `unpack.sh` file (it's in this directory) into the current directory (`/data`). Also find the `./migrate.sh` file that you modified and copy it into the current directory too.

Then just run this script:
```bash
mkdir dbs
cd attachment
../unpack.sh ../dbs
cd ../scope
../unpack.sh ../dbs
cd ../vague
../unpack.sh ../dbs
cd ../dbs
./migrate.sh
```

It should work as is, but again, if one the db's doesn't upload, just delete it (from the questions dataset too).

## Cleanup

You can delete the leftover archives now, the BIRD files take up a lot of space.
