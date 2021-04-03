import numpy as np
import collections
import itertools


class NodeClassifier:
    '''
    Class used for prediction of targets in DecisionMaker, fits/predicts on training/testing data
    '''

    def __init__(self, classifier_object, classifier_mode='internal'):
        self.classifier = classifier_object
        self.classifier_mode = classifier_mode

    def fit(self, df, target):
        self.classifier.fit(df, target)

    def predict(self, df, dm):
        predictions = self.classifier.predict_proba(df)

        # address cases with a missing class in target due to filtration of examples:
        if predictions.shape[1] < dm._class_count:
            pred = np.zeros((predictions.shape[0], dm._class_count))
            pred[:, self.classifier.classes_] = predictions
            predictions = pred

        return predictions

    def _evaluate_q0(self, dm, df, z):
        # extend and mask data for external classifier:
        if self.classifier_mode == 'external':
            mask = [0 if m == 0 else 1 for m in z]

            # override mask by free variables from root of decision maker
            mask = np.where(np.array(dm.z_init) == -1, -1, mask)

            df = dm._extend_mask_df(df, mask)

        # evaluate expected utility from prediction:
        predictions = self.predict(df, dm)

        tc = np.sum(np.where(np.array(z) == 0, 0, dm.costs))
        pred_u = np.matmul(predictions, np.transpose(dm.U))
        q0 = np.max(pred_u, axis=1) - tc

        return q0


class NodePolicy:
    '''
    Class used for selection of costly variables, used by DecisionMaker, fits on observed training data and
    utilities from further DecisionNodes.
    '''

    def __init__(self, policy_class, args):
        self._is_fitted = False
        self.policies = {}
        self.policy_class = policy_class
        self.args = args

    def fit(self, df, target, current_z, node_classifier):
        self.current_z = current_z
        self.z_to_select = [z for z in target]

        # for each adjacent decision node fit a regression model estimating its utility:
        for z in target:
            if z != current_z:
                policy = self.policy_class(**self.args)
                policy.fit(df, target[z])
                self.policies[z] = policy
            else:
                # for current decision node use classifier:
                self.policies[z] = node_classifier

    def _get_q_values(self, df, dm):
        q_values = np.zeros((df.shape[0], len(self.z_to_select)))

        # iterate over adjacent decision nodes and predict expected utility based on available data:
        for z in self.z_to_select:
            indx = self.z_to_select.index(z)
            if z == self.current_z:
                q_values[:, indx] = self.policies[z]._evaluate_q0(dm, df, self.current_z)
            else:
                q_values[:, indx] = self.policies[z].predict(df)

        return q_values

    def return_action(self, df, dm):
        # select adjacent decision node with the highest expected utility:
        q_values = self._get_q_values(df, dm)
        indx = np.argmax(q_values, axis=1)

        return [self.z_to_select[i] for i in indx]

    def _predict_v(self, df, dm):
        q_values = self._get_q_values(df, dm)
        return np.max(q_values, axis=1)


class DecisionNode:
    '''
    Class representing decision nodes. Contains information about currently acquired variables and keeps
    adjacent decision nodes as its children. The decision nodes form DAG of the ACID problem and form
    a key structure for acquisition of variables.
    '''

    def __init__(self, z):
        self.z = z
        self.incoming_z = self._get_incoming_z(z)
        self.outgoing_z = self._get_outgoing_z(z)
        self.children = {}
        self.is_terminal = 0 not in z
        self.prediction_examples = []
        self.training_examples = []
        self._q0 = None
        self.depth = z.count(1)

    def _get_incoming_z(self, z):
        incoming_z = []
        for ind, i in enumerate(z):
            next_z = list(z)
            if i == 1:
                next_z[ind] = 0
                incoming_z.append(tuple(next_z))

        return incoming_z

    def _get_outgoing_z(self, z):
        outgoing_z = []
        for ind, i in enumerate(z):
            next_z = list(z)
            if i == 0:
                next_z[ind] = 1
                outgoing_z.append(tuple(next_z))

        return outgoing_z

    def evaluate(self, df, gold_classes, dm, examples, store_models=True):
        # mask not acquired variables and filter examples in predictors and golden classes:
        z_mask = np.where(np.array(self.z) == 0, False, True)

        if dm.classifier_mode == 'external':
            df = df[examples]
        else:
            df = df[:, z_mask][examples]
        gold_classes = gold_classes[examples]

        # fit internal classifier if applicable:
        if dm.classifier_mode == 'internal':
            node_classifier = NodeClassifier(dm.classifier_class(**dm.classifier_params))
            node_classifier.fit(df, gold_classes)
            if store_models:
                dm.classifier[self.z] = node_classifier

        else:
            node_classifier = dm.classifier

        # get q0 of prediction with currently acquired variables
        self._q0 = node_classifier._evaluate_q0(dm, df, self.z)

        # if node is terminal, use q0 as v
        if self.is_terminal:
            self._v = self._q0
            if store_models:
                dm.policy[self.z] = 'predict'
        # else train policy on v of child nodes
        else:
            policy_targets = {self.z: self._q0}
            for child_z, child in self.children.items():
                policy_targets[child_z] = child._v

            policy = NodePolicy(dm.policy_class, dm.policy_params)
            policy.fit(df, policy_targets, self.z, node_classifier)
            self._v = policy._predict_v(df, dm)

            if store_models:
                dm.policy[self.z] = policy


class DecisionMaker:
    '''
    Main class representing the decision maker. Contains main methods fit, predict and evaluate. Stores trained
    classifier and policy.
    '''

    def __init__(self, depth=-1, classifier_mode='external',
                 classifier_boots=None,
                 classifier_class=None, classifier_params=None,
                 policy_class=None, policy_params=None):
        '''
        :param depth: depth used for search, if -1: exact solution from the last node is calculated, if 1: greedy
                      acquisition of variables is applied
        :param classifier_mode: if 'internal': classifier is build for each decision node,
                                if 'external': external classifier excepting all combinations of missing variables is
                                               is assumed
        :param classifier_boots: specify if classifier_mode == 'external', instance of class NodeClassifier with
                                 an already fitted model
        :param classifier_class: specify if classifier_mode == 'external', class of classification model used for
                                 prediction of target variables
        :param classifier_params: arguments for initialization of instance of class classifier_class
        :param policy_class: class of regression model used for estimation of utility
        :param policy_params: arguments for initialization of instance of class policy_class
        '''

        if classifier_mode == 'external':
            self.classifier = classifier_boots
        elif classifier_mode == 'internal':
            self.classifier = {}
            self.classifier_class = classifier_class
            self.classifier_params = classifier_params

        self.classifier_mode = classifier_mode

        self.policy_class = policy_class
        self.policy_params = policy_params
        self.policy = {}

        self.depth = depth
        self.go_forward_mode = None


    def fit(self, df, gold_classes, costs, U):
        '''
        Trains the decision maker on training data according to parameters set during initialization. Decides what
        variables to acquire (train policy) and train classifier if applicable.
        :param df: Numpy array of features
        :param gold_classes: Numpy array of golden classes
        :param costs: Numpy array of costs of each feature in df
        :param U: Numpy array of utilities for each (mis)classification
        '''

        self.df = df
        self.gold_classes = gold_classes
        self._class_count = np.unique(gold_classes).shape[0]
        self.U = U

        self.costs = costs
        self.z_init = tuple([0 if c > 0 else -1 for c in costs])

        if self.depth == -1:
            self.root = DecisionNode(self.z_init)
            examples = np.arange(df.shape[0])
            self.root = self._evaluate_from_depth(self.root, -1, df, gold_classes, examples)

        else:
            self.go_forward_mode = 'training'
            self.root = DecisionNode(self.z_init)
            self._go_forward(df, gold_classes)

    def predict(self, df):
        '''
        Predict on specified dataframe, selects variables according to fitted policy
        :param df: costly data to predict on
        :return: Numpy array of probabilities for each class and example
        '''
        self.predictions = np.zeros((df.shape[0], self._class_count))
        self.vars_in_prediction = np.zeros(df.shape) # zero one indicator
        self.acquisition_paths = np.zeros(df.shape) - 1 # position of acquired variable
        self._reset(self.root)
        self.go_forward_mode = 'prediction'
        self._go_forward(df, None)

        return self.predictions

    def _reset(self, node):
        # reset prediction examples in nodes to allow next prediction:
        queue = collections.deque()
        queue.append(node)
        while queue:
            node = queue.popleft()
            node.prediction_examples = []
            for z, child in node.children.items():
                if child not in queue:
                    queue.append(child)

    def evaluate(self, gold_classes):
        '''
        Evaluates utility after prediction, has to be run after predict method
        :param gold_classes: gold classes associated with data used for predict function
        :return: Dictionary with total utility, prediction utility and costs
        '''
        tc = np.sum(np.where(np.array(self.vars_in_prediction) == 1, self.costs, 0), axis=1)
        eu = np.matmul(self.predictions, np.transpose(self.U))
        predicted_classes = np.argmax(eu, axis=1)
        pred_u = self.U[predicted_classes, gold_classes]
        u = pred_u - tc

        return {'total_utility': u, 'prediction_utility': pred_u, 'costs': tc}


    def _step_forward(self, node, df, examples):
        # Function returns selected children decision nodes and makes prediction on the node

        # filter examples:
        if self.classifier_mode == 'external':
            df = df[examples]
        else:
            z_mask = np.where(np.array(node.z) == 0, False, True)
            df = df[:, z_mask][examples]

        # for terminal node, perform prediction:
        if self.policy[node.z] == 'predict':
            if self.go_forward_mode == 'prediction':
                if self.classifier_mode == 'external':
                    # mask = [1 if m == 0 else 0 for m in node.z]
                    df = self._extend_mask_df(df, node.z)
                    self.predictions[examples] = self.classifier.predict(df, self)
                else:
                    self.predictions[examples] = self.classifier[node.z].predict(df, self)
                self.vars_in_prediction[examples] = np.array(node.z)

            # return no children as the node is terminal:
            return []

        # for non terminal nodes, select children node or predict:
        else:
            policy = self.policy[node.z]
            z_per_example = policy.return_action(df, self)
            z_unique = list(set(z_per_example))
            children = []

            # iterate over all selected nodes by policy:
            for z in z_unique:
                is_match = [z_ex == z for z_ex in z_per_example]

                # if current node is selected, do prediction
                if tuple(z) == node.z:
                    if self.go_forward_mode == 'prediction':
                        if self.classifier_mode == 'external':
                            # mask = [1 if m == 0 else 0 for m in node.z]
                            df = self._extend_mask_df(df, node.z)
                            self.predictions[examples[is_match]] = self.classifier.predict(df[is_match], self)
                        else:
                            self.predictions[examples[is_match]] = self.classifier[node.z].predict(df[is_match], self)
                        self.vars_in_prediction[examples[is_match]] = np.array(z)
                    continue

                # otherwise select examples for given child:
                child = node.children[z]
                if self.go_forward_mode == 'prediction':
                    child.prediction_examples = np.array(np.append(child.prediction_examples, examples[is_match]),
                                                         dtype=np.int32)
                    action_indx = [i for i, z_i in enumerate(child.z) if node.z[i] != z_i]
                    self.acquisition_paths[:, action_indx[0]][examples[is_match]] = child.depth
                elif self.go_forward_mode == 'training':
                    child.training_examples = np.array(np.append(child.training_examples, examples[is_match]),
                                                         dtype=np.int32)

                children.append(child)

            return children

    def _evaluate_from_depth(self, node_init, depth, df, gold_classes, examples):
        # initialize queue and root z for the estimate from depth
        queue = collections.deque()
        z_init = tuple([-1 if i != 0 else 0 for i in node_init.z])

        # set the leaf nodes for the estimates base on the selected depth:
        if depth == -1:
            z_terminal = tuple([1 if i == 0 else -1 for i in z_init])
            node = DecisionNode(z_terminal)
            queue.append(node)

        elif depth >= 1:
            # limit depth to maximum depth:
            depth = np.minimum(depth, np.sum(np.array(z_init) == 0))

            # generate leaf nodes:
            if depth > 0:
                is_zero_indx = np.where(np.array(z_init) == 0)[0]

                for ind in list(itertools.combinations(is_zero_indx, depth)):
                    z_np = np.array([-1 if i != 0 else 0 for i in z_init])
                    z_np[list(ind)] = 1
                    node = DecisionNode(tuple(z_np))
                    node.is_terminal = True
                    queue.append(node)

        # evaluate decision nodes layer by layer from max depth until reach the root node:
        existing_parents = {z_init: node_init}
        i = 0
        while queue:
            i += 1
            child = queue.popleft()
            print('{} EFD-{}: {}'.format(depth, i, child.z))
            if self.go_forward_mode == 'training':
                store_models = False
            else:
                store_models = True
            child.evaluate(df, gold_classes, self, examples, store_models)

            # append the children to parent, if parent exists do not create a new one:
            for z_previous in child.incoming_z:
                if z_previous in existing_parents:
                    z_child = child.z
                    if z_previous == z_init:
                        z_child = tuple([1 if node_init.z[i] == 1 else z for i, z in enumerate(child.z)])
                    existing_parents[z_previous].children[z_child] = child

                else:
                    parent = DecisionNode(z_previous)
                    parent.children[child.z] = child
                    queue.append(parent)
                    existing_parents[z_previous] = parent

        node_init.evaluate(df, gold_classes, self, examples)
        return node_init

    def _go_forward(self, df, gold_classes):
        # initialize queue and examples:
        examples = np.arange(df.shape[0])
        queue = collections.deque()

        if self.go_forward_mode == 'prediction':
            self.root.prediction_examples = examples
        elif self.go_forward_mode == 'training':
            self.root.training_examples = examples
            visited_training_nodes = {}

        # add root to queue, keep track of visited nodes:
        queue.append(self.root)
        visited_nodes = {self.root.z: self.root}
        i = 0

        # process children in queue layer by layer until it is empty:
        while queue:
            i += 1
            node = queue.popleft()
            print('({}) GF-{}: {}'.format(self.go_forward_mode, i, node.z))
            if self.go_forward_mode == 'training':
                node = self._evaluate_from_depth(node, self.depth, df, gold_classes, node.training_examples)

                # drop children from estimate, generate new children and keep them in dict:
                node.children = {}
                for child_z in node.outgoing_z:
                    if child_z in visited_training_nodes:
                        node.children[child_z] = visited_training_nodes[child_z]
                    else:
                        node.children[child_z] = DecisionNode(child_z)
                        visited_training_nodes[child_z] = node.children[child_z]
                examples = node.training_examples

            elif self.go_forward_mode == 'prediction':
                examples = node.prediction_examples
                examples

            # select next children by step forward:
            children = self._step_forward(node, df, examples)

            # prevent acquisition of variables that are never chosen during training (prediction on node is allowed):
            if self.go_forward_mode == 'training':
                if not type(self.policy[node.z]) is str:
                    self.policy[node.z].z_to_select = [node.z]
                    for child_z in node.outgoing_z:
                        if child_z in [child.z for child in children]:
                            self.policy[node.z].z_to_select.append(child_z)
                        else:
                            self.policy[node.z].policies[child_z] = 'erased'

            # add children to queue if it is not already there:
            for child in children:
                if child.z in visited_nodes:
                    continue
                else:
                    visited_nodes[child.z] = child
                    queue.append(child)

    def _extend_mask_df(self, df, mask):
        # extend and mask dataframe to be compatible with external classifier
        # mask: -1 free var, 0 not acquired, 1 acquired
        is_free = np.where(np.array(mask) == -1, True, False)

        is_free_df = df[:, is_free].copy()
        is_costly_df = df[:, ~is_free].copy()

        mask_costly = [False if m == 1 else True for m in mask if m >= 0]
        mask_df = np.zeros(is_costly_df.shape)
        mask_df[:, mask_costly] = 1

        is_costly_df = np.where(mask_df == 1, 0, is_costly_df)
        final_df = np.concatenate([is_costly_df, mask_df], axis=1)
        final_df = np.concatenate([is_free_df, final_df], axis=1)

        return final_df