#ifndef HASHES_H
#define HASHES_H
#define TABLE_SIZE 50
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

struct sNode;
struct sHashTable;
struct sHastTableVTable;

typedef struct sNode {
	const void *key;
	void *value;
	size_t key_len;
	size_t value_len;
	struct sNode *next;
} tNode;

typedef struct sHastTableVTable {
	bool (*insert_node)(struct sHashTable *self, const void *key, size_t key_len, const void *value, size_t value_len);
	bool (*delete_node)(struct sHashTable *self, const void *key, size_t key_len);
	void* (*get_val)(struct sHashTable *self, const void *key, size_t key_len);
} tHashTableVTable;

typedef struct sHashTable {
	tNode **buckets;
	size_t size;
	size_t count;
	size_t node_val_type_size;
	size_t node_key_type_size;
	const tHashTableVTable* vtable;
} tHashTable;

typedef enum {
	SUCCESS = 0,
	TABLE_CREATE_ERROR = 1,
	DIVBYZERO_INSERT_ERROR = 2,
	INSERT_ERROR = 3,
	RESIZE_ERROR = 4,
	KEY_ERROR = 5,
	SIZE_ERROR = 6,
	DUP_ERROR = 7
} Herrors;

bool insert_node(tHashTable *self, const void *key, size_t key_len, const void *value, size_t value_len);
bool delete_node(tHashTable *self, const void *key, size_t key_len);
void* get_val(tHashTable *self, const void *key, size_t key_len);
void free_table(tHashTable *table);
tHashTable *create_table(size_t size, size_t node_key_type_size, size_t node_val_type_size);

const char* get_error_msg(void);
static inline int8_t* get_err(void);
#define _HERROR (get_err())
#define HERROR *_HERROR

#define HT_INSERT(t, k, kl, v, vl) (t->vtable->insert_node(t, k, kl, v, vl))
#define HT_GET(t, k, kl) (t->vtable->get_val(t, k, kl))
#define HT_DEL(t, k, kl) (t->vtable->delete_node(t, k ,kl))

/* Type Sizes */
#define H_TYPE(T)    sizeof(T)
#define H_PTR        sizeof(void *)
#define H_SIZE       sizeof(size_t)

#define H_CHAR       sizeof(char)
#define H_STR        sizeof(char)
#define H_BOOL       sizeof(bool)

#define H_SHORT      sizeof(short)
#define H_USHORT     sizeof(unsigned short)
#define H_INT        sizeof(int)
#define H_UINT       sizeof(unsigned int)
#define H_LONG       sizeof(long)
#define H_ULONG      sizeof(unsigned long)
#define H_LLONG      sizeof(long long)
#define H_ULLONG     sizeof(unsigned long long)

#define H_FLOAT      sizeof(float)
#define H_DOUBLE     sizeof(double)

#define H_I8         sizeof(int8_t)
#define H_I16        sizeof(int16_t)
#define H_I32        sizeof(int32_t)
#define H_I64        sizeof(int64_t)
#define H_U8         sizeof(uint8_t)
#define H_U16        sizeof(uint16_t)
#define H_U32        sizeof(uint32_t)
#define H_U64        sizeof(uint64_t)

/* Cast Helpers */
#define CAST_GEN(x)     ((void *)(x))
#define CAST_AS(T, x)   (*(T *)(x))
#define CAST_PTR(T, x)  ((T *)(x))

#define CAST_STR(x)     ((char *)(x))
#define CAST_CHAR(x)    (*(char *)(x))
#define CAST_BOOL(x)    (*(bool *)(x))
#define CAST_SIZE(x)    (*(size_t *)(x))

#define CAST_INT(x)     (*(int *)(x))
#define CAST_UINT(x)    (*(unsigned int *)(x))
#define CAST_LONG(x)    (*(long *)(x))
#define CAST_LLONG(x)   (*(long long *)(x))
#define CAST_INT_ARR(x) ((int *)(x))

#define CAST_FLOAT(x)   (*(float *)(x))
#define CAST_DOUBLE(x)  (*(double *)(x))



#ifdef HTABLE_IMPLEMENTATION
static inline int8_t* get_err(void)
{
	static int8_t err = 0;
	return &err;
}

static void *_duplicate_val(const void *data, size_t len, size_t type_size)
{
	if (!data || !len || !type_size) return NULL;
	if (len > SIZE_MAX / type_size) return NULL;
	size_t size_in_bytes = len * type_size;
	uint8_t *dup = malloc(sizeof(uint8_t) * size_in_bytes);
	if (!dup) return NULL;
	return memcpy(dup, data, size_in_bytes);
}

static void _reset_error(void)
{
		int8_t *err = _HERROR;
		*err = 0;
}

static uint32_t _hash(const void *key, size_t key_len_in_bytes, size_t size)
{
	uint8_t *k = (uint8_t *)key;
	uint32_t hash = 0;

	for (size_t i = 0; i < key_len_in_bytes; i++)
		hash = (hash * 31) + k[i];
	return (hash % size);
}

static void _refresh_nodes(tHashTable *table, tNode ***new_buckets, size_t new_size)
{
	size_t i = 0;
	size_t new_index;
	tNode *node;
	tNode *next_node;

	while(i < table->size)
	{
		node = table->buckets[i];
		while (node)
		{
			next_node = node->next;
			new_index = _hash(node->key, (node->key_len * table->node_key_type_size), new_size);
			node->next = (*new_buckets)[new_index];
			(*new_buckets)[new_index] = node;
			node = next_node;
		}
		i++;
	}
}

static bool _resize_table(tHashTable *table)
{
	size_t new_size = table->size * 2;
	tNode **new_buckets = (tNode **)calloc(new_size, sizeof(tNode *));
	if (!new_buckets)
	{
		int8_t *err = _HERROR;
		*err = RESIZE_ERROR;
		return (false);
	}
	_refresh_nodes(table, &new_buckets, new_size);
	free(table->buckets);
	table->buckets = new_buckets;
	table->size = new_size;
	return (true);
}

bool insert_node(tHashTable *self, const void *key, size_t key_len, const void *value, size_t value_len)
{
	if (!self || !key || !key_len || !value || !value_len || !self->size) return false;
	_reset_error();
	uint32_t index;
	tNode *new_node;
	tNode *node;
	double l_factor = (double)self->count / self->size;

	if (l_factor > 0.75)
		if (!_resize_table(self))
			return (false);
	index = _hash(key, (key_len * self->node_key_type_size), self->size);
	node = self->buckets[index];
	while (node)
	{
		if (node->key_len == key_len && !memcmp(node->key, key, (self->node_key_type_size * key_len)))
		{
			void *tmp_val = _duplicate_val(value, value_len, self->node_val_type_size); 
			if (!tmp_val)
			{
				int8_t *err = _HERROR;
				*err = DUP_ERROR;
				return (false);
			}
			free(node->value);
			node->value = tmp_val;
			node->value_len = value_len;
			return (true);
		}
		node = node->next;
	}
	new_node = (tNode *)malloc(sizeof(tNode));
	if (!new_node)
	{
		int8_t *err = _HERROR;
		*err = INSERT_ERROR;
		return (false);
	}
	new_node->key = _duplicate_val(key, key_len, self->node_key_type_size);
	if (!new_node->key)
	{
		free(new_node);
		int8_t *err = _HERROR;
		*err = DUP_ERROR;
		return (false);
	}
	new_node->value = _duplicate_val(value, value_len, self->node_val_type_size);
	if (!new_node->value)
	{
		free((void *)new_node->key);
		free(new_node);
		int8_t *err = _HERROR;
		*err = DUP_ERROR;
		return (false);
	}
	new_node->key_len = key_len;
	new_node->value_len = value_len;
	new_node->next = self->buckets[index];
	self->buckets[index] = new_node;
	self->count++;
	return (true);
}

void* get_val(tHashTable *self, const void *key, size_t key_len)
{
	_reset_error();
	if (!self || !key || !key_len || !self->size) return NULL;
	uint32_t index = _hash(key, (key_len * self->node_key_type_size), self->size);
	tNode* node = self->buckets[index];
	void* value;

	while (node)
	{
		if (node->key_len == key_len && !memcmp(node->key, key, (self->node_key_type_size * key_len)))
		{
			value = node->value;
			return (value);
		}
		node = node->next;
	}
	int8_t *err = _HERROR;
	*err = KEY_ERROR;
	return (NULL);
}

bool delete_node(tHashTable *self, const void *key, size_t key_len)
{
	_reset_error();
	if (!self || !key || !key_len || !self->size) return false;
	uint32_t index = _hash(key, (key_len * self->node_key_type_size), self->size);
	tNode *curr = self->buckets[index];
	tNode *prev = NULL;

	while (curr)
	{
		if (curr->key_len == key_len && !memcmp(key, curr->key, (self->node_key_type_size * key_len)))
		{
			if (prev)
				prev->next = curr->next;
			else
				self->buckets[index] = curr->next;
			self->count--;
			free((void*)curr->key);
			free(curr->value);
			free(curr);
			return (true);
		}
		prev = curr;
		curr = curr->next;
	}
	return (false);
}

tHashTable *create_table(size_t size, size_t node_key_type_size, size_t node_val_type_size)
{
	_reset_error();
	if (!size || !node_key_type_size || !node_val_type_size)
	{
		int8_t *err = _HERROR;
		*err = SIZE_ERROR;
		return (NULL);
	}
	tHashTable *table = (tHashTable *)malloc(sizeof(tHashTable));
	if (!table)
	{
		int8_t *err = _HERROR;
		*err = TABLE_CREATE_ERROR;
		return (NULL);
	}
	static const tHashTableVTable vtable = {insert_node, delete_node, get_val};
	table->vtable = &vtable;
	table->size = size;
	table->buckets = (tNode **)calloc(size, sizeof(tNode *));
	if (!table->buckets)
	{
		free(table);
		int8_t *err = _HERROR;
		*err = TABLE_CREATE_ERROR;
		return (NULL);
	}
	table->count = 0;
	table->node_key_type_size = node_key_type_size;
	table->node_val_type_size = node_val_type_size;
	return (table);
}

void free_table(tHashTable *table)
{
	if (!table) return;
	size_t i = 0;
	tNode *curr;
	tNode *prev;

	while(i < table->size)
	{
		curr = table->buckets[i];
		while (curr)
		{
			prev = curr;
			curr = curr->next;
			free((void*)prev->key);
			free(prev->value);
			free(prev);
		}
		i++;
	}
	free(table->buckets);
	free(table);
}

const char* get_error_msg(void)
{
	int8_t *error = _HERROR;

	switch (*error)
	{
		case SUCCESS: return ("Success");
		case TABLE_CREATE_ERROR: return ("Failed to create the hash table");
		case DIVBYZERO_INSERT_ERROR:
			 return ("Division by zero while inserting, likely cause is table->size = 0");
		case INSERT_ERROR: return ("Failed to insert value to table");
		case RESIZE_ERROR: return ("Failed to resize table");
		case KEY_ERROR: return ("Unknown key passed to get_node()");
		case SIZE_ERROR: return ("Can't create a table with size '0'");
		case DUP_ERROR: return ("Failed to copy a value or key");
		default: return ("Unknown internal system error");
	}
}
#endif


#endif
